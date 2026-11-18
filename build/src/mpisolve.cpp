/*
  Parallelized Finite Difference Time Domain Acoustic / Helmholtz Solver
  Adapted from quantumfdtd_v3 (Michael Strickland, Rafael L. Delgado)
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <cerrno>
#include <complex>
#include <vector>
#include <algorithm>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/types.h>
#include <unistd.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

using namespace std;

#include "mpi.h"
#include "fftw3-mpi.h"
#include "mpisolve.h"
#include "grid.h"
#include "initialconditions.h"
#include "potential.h"
#include "outputroutines.h"
#include "paramreader.h"

// Global parameters initialized from input/params.txt
int NUMX = 20, NUM = 20, STEPS = 40000, UPDATE = 100, SNAPUPDATE = 1000;
int POTENTIAL = 0, INITCONDTYPE = 0, INITCONDAXIS = 0, INITSYMMETRY = 0, SAVEWAVEFNCS = 0, DUMPSNAPS = 0, SNAPDUMP = 0;
double A = 0.05, EPS = 0.001, SIG = 0.06, MASS = 1.0, SIGMA = 0.223, TOLERANCE = -1;
int KINTERM = 0;
char *EXTPOT = NULL;
char *DATAFOLD = NULL;
int SAVEPOT = 0;
int SAVEDECAY = 0;
double POTCRITR = 3., POTFLATR = -1.;
double RHO_OBJECT  = 4.0;     // Relative density (bone/stone/ceramic vs. water)
int    SHARPEN_VDB = 1;       // Steepens blurry VDB transitions so surfaces reflect cleanly

// Acoustic Helmholtz Renderer defaults (overridden by input/params.txt)
double C_MEDIUM    = 1500.0;  // Background wave speed (m/s)
double C_OBJECT    = 950.0;   // Object wave speed (m/s)
double FREQ        = 0.0;     // Frequency in Hz (0 = auto-derive from WAVELENGTH)
double WAVELENGTH  = 4.0;     // Target wavelength in voxels when FREQ <= 0
double SOURCE_AMP  = 1.5;     // Source pressure amplitude
int    BEAM_RADIUS = 20;      // Transducer radius in voxels
int    SPONGE      = 8;       // Absorbing wall thickness in voxels
double SPONGE_DAMP = 0.15;    // Absorbing wall damping factor
char  *VDBFILE     = NULL;    // Input VDB path

// MPI variables
int nodeID, tmpnodeID, numNodes;

// Global step tracker for the continuous acoustic source
int global_step = 0;

// Files
fstream debug_out;
int debug = DEBUG_OFF;

// MPI non-blocking send/receive buffers
double *leftSendBuffer, *rightSendBuffer;
MPI_Status leftSendStatus, rightSendStatus;
MPI_Status leftSendMessageStatus, rightSendMessageStatus;
MPI_Request leftSend, rightSend;
MPI_Request leftMessageSend, rightMessageSend;

double *leftReceiveBuffer, *rightReceiveBuffer;
MPI_Status leftReceiveStatus, rightReceiveStatus;
MPI_Status leftReceiveMessageStatus, rightReceiveMessageStatus;
MPI_Request leftReceive, rightReceive;
MPI_Request leftMessageReceive, rightMessageReceive;

MPI_Comm workers_comm;
MPI_Group workers_group;

// FFTW3 structures
struct fftw_par_t
{
  fftw_plan plan_fftw, plan_ifftw;
  fftw_complex *in, *out;
  ptrdiff_t local_n0, local_0_start;
  ptrdiff_t alloc_local;
} fftw_par;

inline void dcomp_to_fftw(const dcomp &in, fftw_complex *out)
{
  (*out)[0] = in.real();
  (*out)[1] = in.imag();
}
inline void fftw_to_dcomp(const fftw_complex &in, dcomp *out) { *out = dcomp(in[0], in[1]); }
inline double pow2(double a) { return a * a; }
inline dcomp pow2(dcomp a) { return a * a; }

// Observables (kept for link compatibility with outputroutines.cpp)
dcomp energy = 0, energyCollect = 0;
dcomp normalization = 0, normalizationCollect = 0;
dcomp vInfinity = 0, vInfinityCollect = 0;
dcomp rRMS2 = 0, rRMS2Collect = 0;
dcomp xAvg = 0, xAvgCollect = 0;
dcomp yAvg = 0, yAvgCollect = 0;
dcomp zAvg = 0, zAvgCollect = 0;
dcomp EGrnd, timef;
int snapcnt = -1;

int    AUTO_ALIGN  = 1;       // Auto-center and scale input VDB bounding box
double OBJECT_FILL = 0.65;    // Object spans 65% of the NUM grid along its longest axis
int    SOURCE_MODE    = 0;    // 0 = Continuous Wave, 1 = Gaussian Pulse
double PULSE_CYCLES   = 2.5;  // Number of carrier cycles in the tone burst
double PULSE_INTERVAL = 0.0;  // 0.0 = fire once; > 0 = repeat every N wave cycles

int    RECEIVE_ECHO = 0;      // 0 = Off, 1 = Record echoes & reconstruct B-mode
int    RX_STRIDE    = 2;      // Sample every Nth voxel across transducer disc
double BMODE_DB     = 40.0;   // Display dynamic range (dB)
double BMODE_FNUM   = 1.0;    // Receive focal ratio (aperture = depth / FNUM)

struct ReceiverElem {
  int sx, sy, gx, gy;
};
std::vector<ReceiverElem> local_rx;
std::vector<float>        local_rf_data;
int                       rf_num_steps = 0;

void reconstructBMode();

// Shared transform mapping solver voxel indices [0..NUM-1] <-> Houdini World Space
double         vdb_world_step = 0.01;
openvdb::Vec3d vdb_world_origin(0.0, 0.0, 0.0);

void computeObservables(dcomp ***wfnc);
void loadWaveSpeedsFromVDB(const std::string& vdb_filepath);
void exportVDBFrame(int frame_number);

int main(int argc, char *argv[])
{
  int done = 0, checksum = 0;
  char message[1024];
  char fname[1024];
  struct tms starttime, endtime;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &tmpnodeID);
  MPI_Status status;
  MPI_Group all_group;

  nodeID = tmpnodeID + 1;

  MPI_Comm_group(MPI_COMM_WORLD, &all_group);
  MPI_Comm dup_comm_world;
  MPI_Comm_dup(MPI_COMM_WORLD, &dup_comm_world);
  MPI_Comm_group(dup_comm_world, &workers_group);
  MPI_Comm_create(dup_comm_world, workers_group, &workers_comm);

  int workers_rank;
  MPI_Comm_rank(workers_comm, &workers_rank);

  MPI_Barrier(workers_comm);

  cout << "Process nodeID=" << nodeID << "\t workers_rank=" << workers_rank << endl;

  if (debug)
  {
    sprintf(fname, "debug/debug_%d.txt", nodeID);
    debug_out.open(fname, ios::out);
    debug_out << "==> Node " << nodeID << " is ready" << endl;
  }

  if (nodeID == 1)
  {
    times(&starttime);
    print_line();
    cout << "Parameters from file" << endl;
    print_line();
    readParametersFromFile((char *)"input/params.txt", 1);
    if (argc > 1)
    {
      print_line();
      cout << "Parameters from commandline" << endl;
      print_line();
      readParametersFromCommandLine(argc, argv, 1);
    }
  }
  else
  {
    readParametersFromFile((char *)"input/params.txt", 0);
    readParametersFromCommandLine(argc, argv, 0);
  }

  if (!DATAFOLD)
    DATAFOLD = strdup("data");

  if (NUM % numNodes != 0)
  {
    if (nodeID == 1)
      print_line();
    cout << "ERROR: Unable to partition lattice ... exiting" << endl;
    print_line();
    if (debug)
    {
      debug_out << "==> Goodbye from node " << nodeID << endl;
      debug_out.close();
    }
    MPI_Finalize();
    exit(0);
  }
  else
  {
    NUMX = NUM / numNodes;
  }

  if (nodeID == 1)
  {
    for (int node = 1; node < numNodes; node++)
    {
      sprintf(message, "Hello to node %d", node);
      MPI_Send(message, strlen(message), MPI_CHAR, node, HELLO, MPI_COMM_WORLD);
    }

    cout << "Receiving hello from nodes on master node..." << endl;
    checksum = 0;
    while (checksum < numNodes - 1)
    {
      MPI_Recv(&done, 1, MPI_INT, MPI_ANY_SOURCE, HELLO, MPI_COMM_WORLD, &status);
      checksum += done;
    }
    cout << "==> Master node: cluster ready..." << endl;
  }
  else
  {
    MPI_Recv(message, 20, MPI_CHAR, 0, HELLO, MPI_COMM_WORLD, &status);
    done = 1;
    MPI_Send(&done, 1, MPI_INT, 0, HELLO, MPI_COMM_WORLD);
  }

  MPI_Barrier(workers_comm);
  solveInitialize();

  MPI_Barrier(workers_comm);
  solve();

  MPI_Barrier(workers_comm);
  solveFinalize();

  cout.flush();
  MPI_Barrier(workers_comm);

  if (nodeID != 1)
  {
    done = 1;
    MPI_Send(&done, 1, MPI_INT, 0, DONE, MPI_COMM_WORLD);
    if (debug)
    {
      debug_out << "==> Goodbye from node " << nodeID << endl;
      debug_out.close();
    }
  }
  else
  {
    checksum = 0;
    while (checksum < numNodes - 1)
    {
      MPI_Recv(&done, 1, MPI_INT, MPI_ANY_SOURCE, DONE, MPI_COMM_WORLD, &status);
      checksum += done;
      sleep(1.0);
    }

    times(&endtime);
    cout << "==> User time: " << (endtime.tms_utime - starttime.tms_utime) / ((double)sysconf(_SC_CLK_TCK)) << " seconds" << endl;
    cout << "==> System time: " << (endtime.tms_stime - starttime.tms_stime) / ((double)sysconf(_SC_CLK_TCK)) << " seconds" << endl;
    print_line();
    cout << "Done." << endl;
    print_line();
  }

  MPI_Group_free(&workers_group);
  MPI_Finalize();

  return 0;
}

/*---------------------------------------------------------------------------*/
/* OpenVDB Reader & Writer                                                   */
/*---------------------------------------------------------------------------*/

void loadWaveSpeedsFromVDB(const std::string& vdb_filepath)
{
  openvdb::initialize();

  // Default entire local grid (including ghost cells) to background medium speed
  for (int sx = 0; sx < NUMX + 2; sx++) {
    for (int sy = 0; sy < NUM + 2; sy++) {
      for (int sz = 0; sz < NUM + 2; sz++) {
        c_speed[sx][sy][sz] = C_MEDIUM;
        rho_field[sx][sy][sz] = 1.0;
      }
    }
  }

  // Fallback transform if VDB is missing or AUTO_ALIGN == 0
  vdb_world_step = A;
  vdb_world_origin = openvdb::Vec3d(0.0, 0.0, 0.0);

  openvdb::FloatGrid::Ptr grid;
  try {
    openvdb::io::File file(vdb_filepath);
    file.open();
    openvdb::GridBase::Ptr baseGrid = file.readGrid("density");
    file.close();

    grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
    if (!grid) {
      if (nodeID == 1) std::cerr << "Error: VDB grid 'density' is not a FloatGrid!" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  } catch (const openvdb::Exception& e) {
    if (nodeID == 1) {
      std::cerr << "Warning: Could not open " << vdb_filepath << " (" << e.what() 
                << "). Running with uniform medium." << std::endl;
    }
    return;
  }

  // Compute active world-space bounding box of the input VDB
  openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
  if (!bbox.empty() && AUTO_ALIGN != 0) {
    openvdb::Vec3d p0 = grid->indexToWorld(bbox.min());
    openvdb::Vec3d p1 = grid->indexToWorld(bbox.max());

    openvdb::Vec3d minWS(std::min(p0.x(), p1.x()), std::min(p0.y(), p1.y()), std::min(p0.z(), p1.z()));
    openvdb::Vec3d maxWS(std::max(p0.x(), p1.x()), std::max(p0.y(), p1.y()), std::max(p0.z(), p1.z()));

    openvdb::Vec3d centerWS = 0.5 * (minWS + maxWS);
    openvdb::Vec3d extentWS = maxWS - minWS;
    double max_extent = std::max({extentWS.x(), extentWS.y(), extentWS.z()});

    if (max_extent > 1e-9) {
      double clamped_fill = std::max(0.1, std::min(0.9, OBJECT_FILL));
      double target_voxels = clamped_fill * (NUM - 1);

      // World distance per solver voxel
      vdb_world_step = max_extent / target_voxels;

      // Map solver grid center ((NUM-1)/2, (NUM-1)/2, (NUM-1)/2) to VDB centerWS
      double half_grid = 0.5 * (NUM - 1);
      vdb_world_origin = centerWS - vdb_world_step * openvdb::Vec3d(half_grid, half_grid, half_grid);
    }
  }

  openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
  openvdb::tools::GridSampler<openvdb::FloatGrid::Accessor, openvdb::tools::BoxSampler>
      sampler(accessor, grid->transform());

  int local_solid_voxels = 0;
  for (int sx = 0; sx <= NUMX + 1; sx++) {
    for (int sy = 0; sy <= NUM + 1; sy++) {
      for (int sz = 0; sz <= NUM + 1; sz++) {
        int global_x = (nodeID - 1) * NUMX + (sx - 1);
        int global_y = sy - 1;
        int global_z = sz - 1;

        openvdb::Vec3R worldPos(
            vdb_world_origin.x() + global_x * vdb_world_step,
            vdb_world_origin.y() + global_y * vdb_world_step,
            vdb_world_origin.z() + global_z * vdb_world_step
        );

        float density = sampler.wsSample(worldPos);
        density = std::max(0.0f, std::min(1.0f, density));

        if (SHARPEN_VDB) {
          // Steepens blurry upsampled VDB gradients around the 0.5 isosurface
          float t = std::max(0.0f, std::min(1.0f, (density - 0.35f) / 0.30f));
          density = t * t * (3.0f - 2.0f * t);
        }

        if (sx >= 1 && sx <= NUMX && sy >= 1 && sy <= NUM && sz >= 1 && sz <= NUM) {
          if (density > 0.01f) local_solid_voxels++;
        }

        c_speed[sx][sy][sz]   = C_MEDIUM + density * (C_OBJECT - C_MEDIUM);
        rho_field[sx][sy][sz] = 1.0 + density * (RHO_OBJECT - 1.0);
      }
    }
  }

  int total_solid_voxels = 0;
  MPI_Reduce(&local_solid_voxels, &total_solid_voxels, 1, MPI_INT, MPI_SUM, 0, workers_comm);

  if (nodeID == 1) {
    openvdb::Vec3d minWS = grid->indexToWorld(bbox.min());
    openvdb::Vec3d maxWS = grid->indexToWorld(bbox.max());
    openvdb::Vec3d domMax = vdb_world_origin + vdb_world_step * openvdb::Vec3d(NUM, NUM, NUM);

    std::cout << "==> Input VDB World BBox : [" << minWS.x() << ", " << minWS.y() << ", " << minWS.z() 
              << "] to [" << maxWS.x() << ", " << maxWS.y() << ", " << maxWS.z() << "]" << std::endl;
    std::cout << "==> Mapped Solver BBox   : [" << vdb_world_origin.x() << ", " << vdb_world_origin.y() << ", " << vdb_world_origin.z()
              << "] to [" << domMax.x() << ", " << domMax.y() << ", " << domMax.z() << "]" << std::endl;
    std::cout << "==> Solid object voxels sampled across MPI ranks: " << total_solid_voxels << std::endl;
  }
}

void exportVDBFrame(int frame_number)
{
  int local_elements = NUMX * NUM * NUM;
  std::vector<float> local_amp(local_elements);
  std::vector<float> local_helm(local_elements);

  int idx = 0;
  for (int sx = 1; sx <= NUMX; ++sx) {
    for (int sy = 1; sy <= NUM; ++sy) {
      for (int sz = 1; sz <= NUM; ++sz) {
        local_amp[idx]  = static_cast<float>(real(w[sx][sy][sz]));
        // sqrt(2 * <P^2>) recovers the stationary Helmholtz wave envelope |u(x)|
        local_helm[idx] = static_cast<float>(std::sqrt(2.0 * helmholtz_sq[sx][sy][sz]));
        idx++;
      }
    }
  }

  std::vector<float> global_amp;
  std::vector<float> global_helm;
  if (nodeID == 1) {
    global_amp.resize(NUM * NUM * NUM);
    global_helm.resize(NUM * NUM * NUM);
  }

  MPI_Gather(
      local_amp.data(), local_elements, MPI_FLOAT,
      global_amp.data(), local_elements, MPI_FLOAT,
      0, workers_comm
  );

  MPI_Gather(
      local_helm.data(), local_elements, MPI_FLOAT,
      global_helm.data(), local_elements, MPI_FLOAT,
      0, workers_comm
  );

  if (nodeID == 1) {
    // Shared world-space transform (matches auto-aligned input VDB)
    openvdb::math::Transform::Ptr xform =
        openvdb::math::Transform::createLinearTransform(vdb_world_step);
    xform->postTranslate(vdb_world_origin);

    // Grid 1: Transient FDTD Wave Amplitude
    openvdb::FloatGrid::Ptr ampGrid = openvdb::FloatGrid::create(0.0f);
    ampGrid->setTransform(xform);
    ampGrid->setName("amplitude");
    openvdb::FloatGrid::Accessor ampAcc = ampGrid->getAccessor();

    // Grid 2: Steady-State RMS Helmholtz Envelope
    openvdb::FloatGrid::Ptr helmGrid = openvdb::FloatGrid::create(0.0f);
    helmGrid->setTransform(xform->copy());
    helmGrid->setName("helmholtz_energy");
    openvdb::FloatGrid::Accessor helmAcc = helmGrid->getAccessor();

    int g_idx = 0;
    for (int x = 0; x < NUM; ++x) {
      for (int y = 0; y < NUM; ++y) {
        for (int z = 0; z < NUM; ++z) {
          openvdb::Coord ijk(x, y, z);

          float mag = std::abs(global_amp[g_idx]);
          if (mag > 1e-4f) {
            ampAcc.setValue(ijk, mag);
          }

          float h_env = global_helm[g_idx];
          if (h_env > 1e-4f) {
            helmAcc.setValue(ijk, h_env);
          }

          g_idx++;
        }
      }
    }

    char filename[256];
    sprintf(filename, "%s/frame_%04d.vdb", DATAFOLD, frame_number);

    openvdb::io::File file(filename);
    openvdb::GridPtrVec grids;
    grids.push_back(ampGrid);
    grids.push_back(helmGrid);
    file.write(grids);
    file.close();

    std::cout << "==> Exported VDB snapshot (amplitude + helmholtz_energy): " << filename << std::endl;
  }
}
/*---------------------------------------------------------------------------*/
/* Solver Initialization, Loop, and Finalization                             */
/*---------------------------------------------------------------------------*/

void solveInitialize()
{
  allocateMemory();

  if (!VDBFILE)
    VDBFILE = strdup("input/skull_density.vdb");

  // Auto-calculate frequency from voxel wavelength if FREQ is not explicitly set
  if (FREQ <= 0.0) {
    FREQ = C_MEDIUM / (WAVELENGTH * A);
  }

if (nodeID == 1)
  {
    
    double c_max = std::max(C_MEDIUM, C_OBJECT);
    double c_min = std::min(C_MEDIUM, C_OBJECT);
    double cfl_max_dt = A / (c_max * sqrt(3.0));
    double min_voxels_per_wave = (c_min / FREQ) / A;

    mkdir(DATAFOLD, 0777);
    print_line();
    cout << "==> Acoustic Solver Configuration:" << endl;
    cout << "    VDB Geometry File : " << VDBFILE << endl;
    cout << "    Wave Speeds       : Medium = " << C_MEDIUM << " m/s, Object = " << C_OBJECT << " m/s" << endl;
    cout << "    Transducer        : Freq = " << FREQ << " Hz, Amp = " << SOURCE_AMP 
         << ", Radius = " << BEAM_RADIUS << " voxels" << endl;
    cout << "    Wall Absorption   : Sponge = " << SPONGE << " voxels, Damp = " << SPONGE_DAMP << endl;
    cout << "    Resolution Check  : " << min_voxels_per_wave << " voxels/wavelength (min)" << endl;
    cout << "    CFL Stability     : EPS = " << EPS << " s (Max safe EPS = " << cfl_max_dt << " s)" << endl;
    cout << "    Source Mode       : "
         << (SOURCE_MODE == 1 ? "Gaussian Pulse" : "Continuous Wave (CW)") << endl;
    if (SOURCE_MODE == 1) {
      cout << "    Pulse Shape       : " << PULSE_CYCLES << " cycles, Repeat = "
           << PULSE_INTERVAL << " cycles" << endl;
    }

    if (EPS > cfl_max_dt) {
      cerr << "WARNING: EPS exceeds CFL limit! Simulation will explode into NaNs." << endl;
    }
    if (min_voxels_per_wave < 2.5) {
      cerr << "WARNING: Wavelength is under-resolved (< 2.5 voxels). Wave will be evanescent!" << endl;
    }

    double s_min = c_min * EPS / A;
    double sin_arg = sin(M_PI * FREQ * EPS) / s_min;

    if (sin_arg >= 1.0 || min_voxels_per_wave < 4.5) {
      cerr << "WARNING: Wavelength inside object (" << min_voxels_per_wave 
           << " voxels) is below the 2nd-order FDTD dispersion limit (< 4.5 voxels)!" << endl;
    }

    print_line();
    flush(cout);
  }

  loadWaveSpeedsFromVDB(VDBFILE);

  for (int sx = 0; sx < NUMX + 2; sx++) {
    for (int sy = 0; sy < NUM + 2; sy++) {
      for (int sz = 0; sz < NUM + 2; sz++) {
        w[sx][sy][sz] = dcomp(0.0, 0.0);
        W[sx][sy][sz] = dcomp(0.0, 0.0);
        w_prev[sx][sy][sz] = dcomp(0.0, 0.0);
        helmholtz_sq[sx][sy][sz] = 0.0;
      }
    }
  }

  if (RECEIVE_ECHO != 0) {
    rf_num_steps = (STEPS / UPDATE) * UPDATE;
    int stride = std::max(1, RX_STRIDE);
    int center_x = NUM / 2;
    int center_y = NUM / 2;

    local_rx.clear();
    for (int sx = 1; sx <= NUMX; sx++) {
      int gx = (nodeID - 1) * NUMX + (sx - 1);
      int dx = gx - center_x;
      if (dx % stride != 0) continue;

      for (int sy = 1; sy <= NUM; sy++) {
        int gy = sy - 1;
        int dy = gy - center_y;
        if (dy % stride != 0) continue;

        if (dx * dx + dy * dy <= BEAM_RADIUS * BEAM_RADIUS) {
          local_rx.push_back({sx, sy, gx, gy});
        }
      }
    }

    local_rf_data.assign(local_rx.size() * rf_num_steps, 0.0f);

    int local_count = static_cast<int>(local_rx.size());
    int total_count = 0;
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, workers_comm);

    if (nodeID == 1) {
      int z_src = std::max(1, NUM - SPONGE - 1) - 1;
      double min_rt_time = 2.0 * (z_src - SPONGE) * A / C_MEDIUM;
      int recommended_steps = static_cast<int>(min_rt_time / EPS) + 200;

      cout << "==> B-Mode Receiver Array : " << total_count << " hydrophone channels (stride = "
           << stride << "), recording " << rf_num_steps << " steps." << endl;
      cout << "    Full-Depth Echo Budget: ~" << recommended_steps << " steps required." << endl;
      if (SOURCE_MODE == 0) {
        cerr << "WARNING: RECEIVE_ECHO = 1 requires pulsed mode (SOURCE_MODE = 1) for time-of-flight imaging!" << endl;
      }
    }
  }
}

void solve()
{
  int step = 0;
  bool keep = true;

  leftSendBuffer = (double *)malloc(2 * (NUM + 2) * (NUM + 2) * sizeof(double));
  rightSendBuffer = (double *)malloc(2 * (NUM + 2) * (NUM + 2) * sizeof(double));
  leftReceiveBuffer = (double *)malloc(2 * (NUM + 2) * (NUM + 2) * sizeof(double));
  rightReceiveBuffer = (double *)malloc(2 * (NUM + 2) * (NUM + 2) * sizeof(double));

  MPI_Barrier(workers_comm);

  do
  {
    keep = (step <= STEPS - UPDATE);

    if (KINTERM == 0)
      syncBoundaries(w);

    if (step % SNAPUPDATE == 0)
    {
      exportVDBFrame(step / SNAPUPDATE);
    }

    if (keep)
    {
      evolve(UPDATE);
      step += UPDATE;
    }
  } while (keep);

  free(rightSendBuffer);
  free(leftSendBuffer);
  free(rightReceiveBuffer);
  free(leftReceiveBuffer);
}

void solveFinalize()
{
  if (RECEIVE_ECHO != 0 && rf_num_steps > 0)
  {
    reconstructBMode();
  }

  if (KINTERM != 0)
  {
    fftw_destroy_plan(fftw_par.plan_fftw);
    fftw_destroy_plan(fftw_par.plan_ifftw);
    fftw_free(fftw_par.in);
    fftw_free(fftw_par.out);
  }
}

static void writePGM(const std::string& filepath, const std::vector<float>& img, int width, int height)
{
  std::ofstream out(filepath, std::ios::binary);
  if (!out) return;
  out << "P5\n" << width << " " << height << "\n255\n";
  // Row 0 = top of grid (transducer at z = NUM-1), bottom row = deep z = 0
  for (int row = 0; row < height; ++row) {
    int gz = (height - 1) - row;
    for (int col = 0; col < width; ++col) {
      float v = std::max(0.0f, std::min(1.0f, img[col * height + gz]));
      unsigned char px = static_cast<unsigned char>(v * 255.0f + 0.5f);
      out.write(reinterpret_cast<const char*>(&px), 1);
    }
  }
}

void reconstructBMode()
{
  int local_count = static_cast<int>(local_rx.size());
  std::vector<int> rx_counts(numNodes, 0);
  MPI_Gather(&local_count, 1, MPI_INT, rx_counts.data(), 1, MPI_INT, 0, workers_comm);

  std::vector<int> coord_counts(numNodes, 0), coord_displs(numNodes, 0);
  std::vector<int> rf_counts(numNodes, 0), rf_displs(numNodes, 0);
  int total_rx = 0;

  if (nodeID == 1) {
    for (int r = 0; r < numNodes; ++r) {
      coord_counts[r] = 2 * rx_counts[r];
      coord_displs[r] = 2 * total_rx;
      rf_counts[r]    = rx_counts[r] * rf_num_steps;
      rf_displs[r]    = total_rx * rf_num_steps;
      total_rx       += rx_counts[r];
    }
  }

  std::vector<int> local_coords(2 * local_count);
  for (int m = 0; m < local_count; ++m) {
    local_coords[2 * m]     = local_rx[m].gx;
    local_coords[2 * m + 1] = local_rx[m].gy;
  }

  std::vector<int>   global_coords(nodeID == 1 ? 2 * total_rx : 1);
  std::vector<float> global_rf(nodeID == 1 ? std::max(1, total_rx * rf_num_steps) : 1);
  int   dummy_int   = 0;
  float dummy_float = 0.0f;

  MPI_Gatherv(
      local_coords.empty() ? &dummy_int : local_coords.data(), 2 * local_count, MPI_INT,
      global_coords.data(), coord_counts.data(), coord_displs.data(), MPI_INT,
      0, workers_comm
  );

  MPI_Gatherv(
      local_rf_data.empty() ? &dummy_float : local_rf_data.data(), local_count * rf_num_steps, MPI_FLOAT,
      global_rf.data(), rf_counts.data(), rf_displs.data(), MPI_FLOAT,
      0, workers_comm
  );

  if (nodeID != 1 || total_rx == 0) return;

  std::cout << "==> Reconstructing 2D Delay-and-Sum B-Mode Ultrasound (" << total_rx << " channels)..." << std::endl;

  const int N_t = rf_num_steps;
  double tau = std::max(0.5, PULSE_CYCLES) / (2.0 * FREQ);
  double t0  = 3.5 * tau;
  double mute_t     = 2.4 * t0;
  double mute_width = 0.2 * t0;

  // 1. Bandpass + Hilbert Analytic Signal via 1D FFTW
  std::vector<std::complex<float>> analytic_rf(static_cast<size_t>(total_rx) * N_t);
  fftw_complex* fft_in  = fftw_alloc_complex(N_t);
  fftw_complex* fft_mid = fftw_alloc_complex(N_t);
  fftw_complex* fft_out = fftw_alloc_complex(N_t);

  fftw_plan p_fwd = fftw_plan_dft_1d(N_t, fft_in, fft_mid, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_plan p_inv = fftw_plan_dft_1d(N_t, fft_mid, fft_out, FFTW_BACKWARD, FFTW_ESTIMATE);

  for (int m = 0; m < total_rx; ++m) {
    const float* trace = &global_rf[static_cast<size_t>(m) * N_t];
    for (int n = 0; n < N_t; ++n) {
      double t = (n + 1) * EPS;
      double tx_gate = 0.5 * (1.0 + std::tanh((t - mute_t) / mute_width));
      fft_in[n][0] = trace[n] * tx_gate;
      fft_in[n][1] = 0.0;
    }

    fftw_execute(p_fwd);

    for (int k = 0; k < N_t; ++k) {
      double h_mult = 0.0;
      if (k > 0 && k < (N_t + 1) / 2) {
        h_mult = 2.0;
      } else if (N_t % 2 == 0 && k == N_t / 2) {
        h_mult = 1.0;
      }

      if (h_mult > 0.0) {
        double freq_k = static_cast<double>(k) / (N_t * EPS);
        double bp = std::exp(-0.5 * pow2((freq_k - FREQ) / (0.65 * FREQ)));
        double scale = (h_mult * bp) / static_cast<double>(N_t);
        fft_mid[k][0] *= scale;
        fft_mid[k][1] *= scale;
      } else {
        fft_mid[k][0] = 0.0;
        fft_mid[k][1] = 0.0;
      }
    }

    fftw_execute(p_inv);

    std::complex<float>* dst = &analytic_rf[static_cast<size_t>(m) * N_t];
    for (int n = 0; n < N_t; ++n) {
      dst[n] = std::complex<float>(static_cast<float>(fft_out[n][0]),
                                   static_cast<float>(fft_out[n][1]));
    }
  }

  fftw_destroy_plan(p_fwd);
  fftw_destroy_plan(p_inv);
  fftw_free(fft_in);
  fftw_free(fft_mid);
  fftw_free(fft_out);

  // 2. Calibrate 2nd-order FDTD discrete group velocity
  double omega = 2.0 * M_PI * FREQ;
  double s_med = C_MEDIUM * EPS / A;
  double v_eff = C_MEDIUM;
  double sin_ratio = std::sin(0.5 * omega * EPS) / s_med;
  if (sin_ratio > 0.0 && sin_ratio < 0.995) {
    double kA_half = std::asin(sin_ratio);
    v_eff = C_MEDIUM * std::cos(kA_half) / std::cos(0.5 * omega * EPS);
  }

  // 3. Plane-Wave Transmit + Dynamic-Receive Delay-and-Sum (XZ and YZ Biplane)
  int z_src    = std::max(1, NUM - SPONGE - 1) - 1;
  int center_x = NUM / 2;
  int center_y = NUM / 2;
  double fnum  = std::max(0.25, BMODE_FNUM);

  auto beamformSlice = [&](bool is_xz, std::vector<float>& env_slice) {
    env_slice.assign(NUM * NUM, 0.0f);
    for (int u = 0; u < NUM; ++u) {
      int px = is_xz ? u : center_x;
      int py = is_xz ? center_y : u;

      for (int gz = 0; gz < z_src - 2; ++gz) {
        double dz = (z_src - gz) * A;
        double half_ap  = std::max(2.0 * A, dz / (2.0 * fnum));
        double half_ap2 = half_ap * half_ap;

        std::complex<double> acc(0.0, 0.0);
        double w_sum = 0.0;

        for (int m = 0; m < total_rx; ++m) {
          double dx = (px - global_coords[2 * m]) * A;
          double dy = (py - global_coords[2 * m + 1]) * A;
          double r_lat2 = dx * dx + dy * dy;
          if (r_lat2 >= half_ap2) continue;

          double r_lat = std::sqrt(r_lat2);
          double apod  = 0.5 * (1.0 + std::cos(M_PI * r_lat / half_ap));
          double d_rx  = std::sqrt(r_lat2 + dz * dz);

          double t_echo = t0 + (dz + d_rx) / v_eff;
          double f_idx  = (t_echo / EPS) - 1.0;

          if (f_idx >= 0.0 && f_idx < N_t - 1) {
            int idx0 = static_cast<int>(f_idx);
            float frac = static_cast<float>(f_idx - idx0);
            const std::complex<float>* ch = &analytic_rf[static_cast<size_t>(m) * N_t];
            std::complex<float> s = (1.0f - frac) * ch[idx0] + frac * ch[idx0 + 1];

            double tgc = d_rx / A; // 3D spherical spreading Time-Gain Compensation
            acc   += apod * tgc * std::complex<double>(s.real(), s.imag());
            w_sum += apod;
          }
        }

        if (w_sum > 1e-6) {
          env_slice[u * NUM + gz] = static_cast<float>(std::abs(acc) / w_sum);
        }
      }
    }
  };

  std::vector<float> env_xz, env_yz;
  beamformSlice(true,  env_xz);
  beamformSlice(false, env_yz);

  // 4. Logarithmic dB Compression
  float max_env = 1e-12f;
  for (float v : env_xz) max_env = std::max(max_env, v);
  for (float v : env_yz) max_env = std::max(max_env, v);

  float dyn_db = static_cast<float>(std::max(10.0, BMODE_DB));
  auto toLogBMode = [&](std::vector<float>& img) {
    for (float& v : img) {
      float norm = std::max(v / max_env, 1e-6f);
      float db   = 20.0f * std::log10(norm);
      v = std::max(0.0f, std::min(1.0f, (db + dyn_db) / dyn_db));
    }
  };
  toLogBMode(env_xz);
  toLogBMode(env_yz);

  // 5. Write 2D PGM Images and World-Space Aligned OpenVDB Biplane Slices
  std::string pgm_xz = std::string(DATAFOLD) + "/bmode_xz.pgm";
  std::string pgm_yz = std::string(DATAFOLD) + "/bmode_yz.pgm";
  writePGM(pgm_xz, env_xz, NUM, NUM);
  writePGM(pgm_yz, env_yz, NUM, NUM);

  openvdb::math::Transform::Ptr xform =
      openvdb::math::Transform::createLinearTransform(vdb_world_step);
  xform->postTranslate(vdb_world_origin);

  openvdb::FloatGrid::Ptr xzGrid = openvdb::FloatGrid::create(0.0f);
  xzGrid->setTransform(xform);
  xzGrid->setName("bmode_xz");
  openvdb::FloatGrid::Accessor xzAcc = xzGrid->getAccessor();

  openvdb::FloatGrid::Ptr yzGrid = openvdb::FloatGrid::create(0.0f);
  yzGrid->setTransform(xform->copy());
  yzGrid->setName("bmode_yz");
  openvdb::FloatGrid::Accessor yzAcc = yzGrid->getAccessor();

  // Populate 3-voxel-thick orthogonal slabs so Houdini raymarching renders them crisply
  for (int u = 0; u < NUM; ++u) {
    for (int gz = 0; gz < NUM; ++gz) {
      float v_xz = env_xz[u * NUM + gz];
      if (v_xz > 1e-3f) {
        for (int dy = -1; dy <= 1; ++dy) {
          xzAcc.setValue(openvdb::Coord(u, center_y + dy, gz), v_xz);
        }
      }
      float v_yz = env_yz[u * NUM + gz];
      if (v_yz > 1e-3f) {
        for (int dx = -1; dx <= 1; ++dx) {
          yzAcc.setValue(openvdb::Coord(center_x + dx, u, gz), v_yz);
        }
      }
    }
  }

  std::string vdb_path = std::string(DATAFOLD) + "/bmode_reconstruction.vdb";
  openvdb::io::File vdb_file(vdb_path);
  openvdb::GridPtrVec grids = {xzGrid, yzGrid};
  vdb_file.write(grids);
  vdb_file.close();

  std::cout << "==> Saved 2D B-Mode images: " << pgm_xz << ", " << pgm_yz << std::endl;
  std::cout << "==> Saved 3D-aligned B-Mode VDB: " << vdb_path << std::endl;
}

void evolve(int nsteps)
{
  int leftTest, rightTest;

  for (int i = 1; i <= nsteps; i++)
  {
    global_step++;

    if (KINTERM == 0)
    {
      leftTest = 1;
      rightTest = 1;
      if (i > 1)
      {
        if (nodeID < numNodes)
        {
          receiveRightBoundary();
          rightTest = 0;
        }
        if (nodeID - 1 >= 1)
        {
          receiveLeftBoundary();
          leftTest = 0;
        }
        while (!leftTest || !rightTest)
        {
          if (!rightTest)
          {
            MPI_Test(&rightReceive, &rightTest, &rightReceiveStatus);
            if (rightTest)
              loadRightBoundaryFromBuffer(w);
          }
          if (!leftTest)
          {
            MPI_Test(&leftReceive, &leftTest, &leftReceiveStatus);
            if (leftTest)
              loadLeftBoundaryFromBuffer(w);
          }
        }
      }
    }

    // Compute c^2 * Laplacian(w)
    wfncKinetic(w);

    // Update slice boundaries
    updateBoundaries(EPS);

    if (KINTERM == 0)
    {
      if (i > 1)
      {
        if (nodeID < numNodes)
          MPI_Wait(&rightSend, &rightSendStatus);
        if (nodeID > 1)
          MPI_Wait(&leftSend, &leftSendStatus);
      }

      leftTest = 1;
      rightTest = 1;
      if (i == 1)
      {
        if (nodeID < numNodes)
          sendRightBoundary(W);
        if (nodeID > 1)
          sendLeftBoundary(W);
      }
      else if (i != nsteps && i > 1)
      {
        if (nodeID < numNodes)
          rightTest = 0;
        if (nodeID > 1)
          leftTest = 0;
        while (!leftTest || !rightTest)
        {
          if (!rightTest)
          {
            MPI_Test(&rightSend, &rightTest, &rightSendStatus);
            if (rightTest)
              sendRightBoundary(W);
          }
          if (!leftTest)
          {
            MPI_Test(&leftSend, &leftTest, &leftSendStatus);
            if (leftTest)
              sendLeftBoundary(W);
          }
        }
      }
    }

    // Update interior cells via 2nd-order wave equation
    updateInterior(EPS);

    // 1. Configurable Absorbing Boundary Sponge Layer
    if (SPONGE > 0) {
      for (int sx = 1; sx <= NUMX; sx++) {
        int gx = (nodeID - 1) * NUMX + (sx - 1);
        for (int sy = 1; sy <= NUM; sy++) {
          int gy = sy - 1;
          for (int sz = 1; sz <= NUM; sz++) {
            int gz = sz - 1;

            int d_min = std::min({gx, NUM - 1 - gx, gy, NUM - 1 - gy, gz, NUM - 1 - gz});
            if (d_min < SPONGE) {
              double norm_d = static_cast<double>(SPONGE - d_min) / SPONGE;
              double damp = exp(-SPONGE_DAMP * norm_d * norm_d);
              W[sx][sy][sz] *= damp;
              w[sx][sy][sz] *= damp;
            }
          }
        }
      }
    }

    // 2. Configurable Disc Transducer Source (CW vs. Gated Gaussian Pulse)
    double current_time = global_step * EPS;
    double signal = 0.0;
    bool source_active = true;

    if (SOURCE_MODE == 0) {
      // Continuous Wave (CW)
      signal = SOURCE_AMP * sin(2.0 * M_PI * FREQ * current_time);
    } else {
      // Gaussian-modulated tone burst: exp(-(u / tau)^2) * sin(2 * pi * f * u)
      double tau = std::max(0.5, PULSE_CYCLES) / (2.0 * FREQ);
      double t0  = 3.5 * tau; // Delay peak so envelope starts smoothly near 0 at t=0

      double local_t = current_time;
      if (PULSE_INTERVAL > 0.0) {
        double rep_period = PULSE_INTERVAL / FREQ;
        if (rep_period > 2.0 * t0) {
          local_t = fmod(current_time, rep_period);
        }
      }

      double u = local_t - t0;
      double envelope = exp(-(u * u) / (tau * tau));

      if (envelope > 1e-4) {
        signal = SOURCE_AMP * envelope * sin(2.0 * M_PI * FREQ * u);
      } else {
        // Release transducer voxels so returning echoes pass through without reflecting
        source_active = false;
      }
    }

    if (source_active) {
      int center_x = NUM / 2;
      int center_y = NUM / 2;
      int sz = std::max(1, NUM - SPONGE - 1);

      for (int sx = 1; sx <= NUMX; sx++) {
        int global_x = (nodeID - 1) * NUMX + (sx - 1);
        int dx = global_x - center_x;
        for (int sy = 1; sy <= NUM; sy++) {
          int dy = (sy - 1) - center_y;
          if (dx * dx + dy * dy <= BEAM_RADIUS * BEAM_RADIUS) {
            W[sx][sy][sz] = dcomp(signal, 0.0);
          }
        }
      }
    }

// Advance time planes: w_prev = w, w = W
    copyDown();

    // Record returning echoes at the transducer surface
    if (RECEIVE_ECHO != 0) {
      int t_idx = global_step - 1;
      if (t_idx >= 0 && t_idx < rf_num_steps) {
        int sz_rx = std::max(1, NUM - SPONGE - 1);
        for (size_t m = 0; m < local_rx.size(); ++m) {
          local_rf_data[m * rf_num_steps + t_idx] =
              static_cast<float>(real(w[local_rx[m].sx][local_rx[m].sy][sz_rx]));
        }
      }
    }

    // 3. Accumulate Envelope (4 cycles for CW standing wave, 1 cycle for B-mode pulse envelope)
    double window_cycles = (SOURCE_MODE == 1) ? 1.0 : 4.0;
    double alpha = std::min(1.0, (EPS * FREQ) / window_cycles);
    for (int sx = 1; sx <= NUMX; sx++) {
      for (int sy = 1; sy <= NUM; sy++) {
        for (int sz = 1; sz <= NUM; sz++) {
          double p = real(w[sx][sy][sz]);
          helmholtz_sq[sx][sy][sz] = (1.0 - alpha) * helmholtz_sq[sx][sy][sz] + alpha * (p * p);
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/* Boundary Sync & Spatial Derivatives                                       */
/*---------------------------------------------------------------------------*/

void syncBoundaries(dcomp ***wfnc)
{
  if (nodeID < numNodes)
  {
    sendRightBoundary(wfnc);
    receiveRightBoundary();
  }
  if (nodeID > 1)
  {
    sendLeftBoundary(wfnc);
    receiveLeftBoundary();
  }

  if (nodeID < numNodes)
  {
    MPI_Wait(&rightReceive, &rightReceiveStatus);
    loadRightBoundaryFromBuffer(wfnc);
    MPI_Wait(&rightSend, &rightSendStatus);
  }
  if (nodeID > 1)
  {
    MPI_Wait(&leftReceive, &leftReceiveStatus);
    loadLeftBoundaryFromBuffer(wfnc);
    MPI_Wait(&leftSend, &leftSendStatus);
  }
}

void sendRightBoundary(dcomp ***wfnc)
{
  for (int sy = 0; sy < NUM + 2; sy++)
    for (int sz = 0; sz < NUM + 2; sz++)
    {
      rightSendBuffer[sy * (NUM + 2) + sz % (NUM + 2)] = real(wfnc[NUMX][sy][sz]);
      rightSendBuffer[sy * (NUM + 2) + sz % (NUM + 2) + (NUM + 2) * (NUM + 2)] = imag(wfnc[NUMX][sy][sz]);
    }
  MPI_Isend(rightSendBuffer, 2 * (NUM + 2) * (NUM + 2), MPI_DOUBLE, nodeID, SYNC_RIGHT, MPI_COMM_WORLD, &rightSend);
}

void sendLeftBoundary(dcomp ***wfnc)
{
  for (int sy = 0; sy < NUM + 2; sy++)
    for (int sz = 0; sz < NUM + 2; sz++)
    {
      leftSendBuffer[sy * (NUM + 2) + sz % (NUM + 2)] = real(wfnc[1][sy][sz]);
      leftSendBuffer[sy * (NUM + 2) + sz % (NUM + 2) + (NUM + 2) * (NUM + 2)] = imag(wfnc[1][sy][sz]);
    }
  MPI_Isend(leftSendBuffer, 2 * (NUM + 2) * (NUM + 2), MPI_DOUBLE, nodeID - 2, SYNC_LEFT, MPI_COMM_WORLD, &leftSend);
}

void receiveRightBoundary()
{
  MPI_Irecv(rightReceiveBuffer, 2 * (NUM + 2) * (NUM + 2), MPI_DOUBLE, nodeID, SYNC_LEFT, MPI_COMM_WORLD, &rightReceive);
}

inline void loadRightBoundaryFromBuffer(dcomp ***wfnc)
{
  for (int sy = 0; sy < NUM + 2; sy++)
    for (int sz = 0; sz < NUM + 2; sz++)
      wfnc[NUMX + 1][sy][sz] = dcomp(rightReceiveBuffer[sy * (NUM + 2) + sz % (NUM + 2)], rightReceiveBuffer[sy * (NUM + 2) + sz % (NUM + 2) + (NUM + 2) * (NUM + 2)]);
}

void receiveLeftBoundary()
{
  MPI_Irecv(leftReceiveBuffer, 2 * (NUM + 2) * (NUM + 2), MPI_DOUBLE, nodeID - 2, SYNC_RIGHT, MPI_COMM_WORLD, &leftReceive);
}

inline void loadLeftBoundaryFromBuffer(dcomp ***wfnc)
{
  for (int sy = 0; sy < NUM + 2; sy++)
    for (int sz = 0; sz < NUM + 2; sz++)
      wfnc[0][sy][sz] = dcomp(leftReceiveBuffer[sy * (NUM + 2) + sz % (NUM + 2)], leftReceiveBuffer[sy * (NUM + 2) + sz % (NUM + 2) + (NUM + 2) * (NUM + 2)]);
}

void wfncKinetic(dcomp ***wfnc)
{
  switch (KINTERM)
  {
  case 0: // VARIABLE-DENSITY ACOUSTIC OPERATOR: rho * c^2 * div((1/rho) * grad(P))
    {
      double inv_A2 = 1.0 / (A * A);
      for (int sx = 1; sx <= NUMX; sx++) {
        for (int sy = 1; sy <= NUM; sy++) {
          for (int sz = 1; sz <= NUM; sz++) {
            double rc = rho_field[sx][sy][sz];
            double local_c = c_speed[sx][sy][sz];
            double bulk = rc * local_c * local_c;

            // Harmonic inverse-density face weights: 2 / (rho_i + rho_j)
            double bx_p = 2.0 / (rc + rho_field[sx + 1][sy][sz]);
            double bx_m = 2.0 / (rc + rho_field[sx - 1][sy][sz]);
            double by_p = 2.0 / (rc + rho_field[sx][sy + 1][sz]);
            double by_m = 2.0 / (rc + rho_field[sx][sy - 1][sz]);
            double bz_p = 2.0 / (rc + rho_field[sx][sy][sz + 1]);
            double bz_m = 2.0 / (rc + rho_field[sx][sy][sz - 1]);

            double p_c = real(wfnc[sx][sy][sz]);
            double div_grad = (
                bx_p * (real(wfnc[sx + 1][sy][sz]) - p_c) - bx_m * (p_c - real(wfnc[sx - 1][sy][sz])) +
                by_p * (real(wfnc[sx][sy + 1][sz]) - p_c) - by_m * (p_c - real(wfnc[sx][sy - 1][sz])) +
                bz_p * (real(wfnc[sx][sy][sz + 1]) - p_c) - bz_m * (p_c - real(wfnc[sx][sy][sz - 1]))
            ) * inv_A2;

            t_kin[sx][sy][sz] = dcomp(bulk * div_grad, 0.0);
          }
        }
      }
    }
    break;

  default:
    cerr << " >> FATAL ERROR: Set KINTERM = 0 for FDTD acoustic solver." << endl;
    exit(1);
  }
}

// Stubs retained if referenced by other headers
void computeObservables(dcomp ***wfnc) {}
void findExcitedStates(const double time, int step) {}
dcomp computeOverlap(dcomp ***wfnc1, dcomp ***wfnc2) { return dcomp(0.0, 0.0); }