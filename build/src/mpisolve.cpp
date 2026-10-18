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

  const double C_AIR = 343.0;
  const double C_BONE = 3000.0;

  // Default entire local grid (including ghost cells) to air speed
  for (int sx = 0; sx < NUMX + 2; sx++)
    for (int sy = 0; sy < NUM + 2; sy++)
      for (int sz = 0; sz < NUM + 2; sz++)
        c_speed[sx][sy][sz] = C_AIR;

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
                << "). Running with uniform air medium." << std::endl;
    }
    return;
  }

  openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
  openvdb::tools::GridSampler<openvdb::FloatGrid::Accessor, openvdb::tools::BoxSampler>
      sampler(accessor, grid->transform());

  for (int sx = 1; sx <= NUMX; sx++) {
    for (int sy = 1; sy <= NUM; sy++) {
      for (int sz = 1; sz <= NUM; sz++) {
        int global_x = (nodeID - 1) * NUMX + (sx - 1);
        int global_y = sy - 1;
        int global_z = sz - 1;

        openvdb::Vec3R worldPos(global_x * A, global_y * A, global_z * A);
        float density = sampler.wsSample(worldPos);
        density = std::max(0.0f, std::min(1.0f, density));

        c_speed[sx][sy][sz] = C_AIR + density * (C_BONE - C_AIR);
      }
    }
  }

  if (nodeID == 1) {
    std::cout << "==> Successfully mapped OpenVDB density to MPI wave speeds." << std::endl;
  }
  int local_solid_voxels = 0;
  for (int sx = 1; sx <= NUMX; sx++) {
    for (int sy = 1; sy <= NUM; sy++) {
      for (int sz = 1; sz <= NUM; sz++) {
        int global_x = (nodeID - 1) * NUMX + (sx - 1);
        int global_y = sy - 1;
        int global_z = sz - 1;

        openvdb::Vec3R worldPos(global_x * A, global_y * A, global_z * A);
        float density = sampler.wsSample(worldPos);
        density = std::max(0.0f, std::min(1.0f, density));

        if (density > 0.01f) local_solid_voxels++;
        c_speed[sx][sy][sz] = C_AIR + density * (C_BONE - C_AIR);
      }
    }
  }

  int total_solid_voxels = 0;
  MPI_Reduce(&local_solid_voxels, &total_solid_voxels, 1, MPI_INT, MPI_SUM, 0, workers_comm);

  if (nodeID == 1) {
    openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
    openvdb::Vec3d minWS = grid->indexToWorld(bbox.min());
    openvdb::Vec3d maxWS = grid->indexToWorld(bbox.max());
    std::cout << "==> VDB World BBox: [" << minWS.x() << ", " << minWS.y() << ", " << minWS.z() 
              << "] to [" << maxWS.x() << ", " << maxWS.y() << ", " << maxWS.z() << "]" << std::endl;
    std::cout << "==> Solver Grid BBox: [0, 0, 0] to [" 
              << NUM * A << ", " << NUM * A << ", " << NUM * A << "]" << std::endl;
    std::cout << "==> Solid teapot voxels sampled across MPI ranks: " << total_solid_voxels << std::endl;
  }
}

void exportVDBFrame(int frame_number)
{
  int local_elements = NUMX * NUM * NUM;
  std::vector<float> local_slice(local_elements);

  int idx = 0;
  for (int sx = 1; sx <= NUMX; ++sx) {
    for (int sy = 1; sy <= NUM; ++sy) {
      for (int sz = 1; sz <= NUM; ++sz) {
        local_slice[idx++] = static_cast<float>(real(w[sx][sy][sz]));
      }
    }
  }

  std::vector<float> global_data;
  if (nodeID == 1) {
    global_data.resize(NUM * NUM * NUM);
  }

  MPI_Gather(
      local_slice.data(), local_elements, MPI_FLOAT,
      global_data.data(), local_elements, MPI_FLOAT,
      0, workers_comm
  );

  if (nodeID == 1) {
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(A));
    grid->setName("amplitude");

    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

    int g_idx = 0;
    for (int x = 0; x < NUM; ++x) {
      for (int y = 0; y < NUM; ++y) {
        for (int z = 0; z < NUM; ++z) {
          // Store rectified magnitude so both positive and negative wave phases glow in Houdini
          float mag = std::abs(global_data[g_idx++]);
          if (mag > 1e-4f) {
            accessor.setValue(openvdb::Coord(x, y, z), mag);
          }
        }
      }
    }

    char filename[256];
    sprintf(filename, "%s/frame_%04d.vdb", DATAFOLD, frame_number);

    openvdb::io::File file(filename);
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    file.write(grids);
    file.close();

    std::cout << "==> Exported VDB snapshot: " << filename << std::endl;
  }
}

/*---------------------------------------------------------------------------*/
/* Solver Initialization, Loop, and Finalization                             */
/*---------------------------------------------------------------------------*/

void solveInitialize()
{
  // Allocate 3D grids (including c_speed and w_prev in grid.cpp)
  allocateMemory();

  if (nodeID == 1)
  {
    print_line();
    cout << "==> Loading OpenVDB Wave Speed Medium" << endl;
    flush(cout);
  }
  loadWaveSpeedsFromVDB("input/skull_density.vdb");

  // Initialize acoustic pressure fields (w, W, w_prev) to silence (0.0)
  for (int sx = 0; sx < NUMX + 2; sx++) {
    for (int sy = 0; sy < NUM + 2; sy++) {
      for (int sz = 0; sz < NUM + 2; sz++) {
        w[sx][sy][sz] = dcomp(0.0, 0.0);
        W[sx][sy][sz] = dcomp(0.0, 0.0);
        w_prev[sx][sy][sz] = dcomp(0.0, 0.0);
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
  if (KINTERM != 0)
  {
    fftw_destroy_plan(fftw_par.plan_fftw);
    fftw_destroy_plan(fftw_par.plan_ifftw);
    fftw_free(fftw_par.in);
    fftw_free(fftw_par.out);
  }
}

void evolve(int nsteps)
{
  int leftTest, rightTest;
  const double FREQ = 343.0 / (8.0 * A);

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

    // Wavelength locked to 8 voxels in air so it always resolves cleanly on the grid
    const double wavelength = 8.0 * A;
    const double FREQ = 343.0 / wavelength;
    const double SOURCE_AMP = 10.0;
    const int DISC_RADIUS = 5;

    double current_time = global_step * EPS;
    double signal = SOURCE_AMP * sin(2.0 * M_PI * FREQ * current_time);

    int center_x = NUM / 2;
    int center_y = NUM / 2;
    int sz = NUM - 3; // Near top of Z-axis

    for (int sx = 1; sx <= NUMX; sx++) {
      int global_x = (nodeID - 1) * NUMX + (sx - 1);
      int dx = global_x - center_x;
      for (int sy = 1; sy <= NUM; sy++) {
        int dy = (sy - 1) - center_y;
        if (dx * dx + dy * dy <= DISC_RADIUS * DISC_RADIUS) {
          W[sx][sy][sz] = dcomp(signal, 0.0);
        }
      }
    }

    // Advance time planes: w_prev = w, w = W
    copyDown();
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
  case 0: // 3D ACOUSTIC WAVE EQUATION SPATIAL OPERATOR: c(x,y,z)^2 * Laplacian(P)
    for (int sx = 1; sx <= NUMX; sx++) {
      for (int sy = 1; sy <= NUM; sy++) {
        for (int sz = 1; sz <= NUM; sz++) {
          double local_c = c_speed[sx][sy][sz];
          double c_sq = local_c * local_c;

          dcomp laplacian = (wfnc[sx + 1][sy][sz] + wfnc[sx - 1][sy][sz] +
                             wfnc[sx][sy + 1][sz] + wfnc[sx][sy - 1][sz] +
                             wfnc[sx][sy][sz + 1] + wfnc[sx][sy][sz - 1] -
                             ((dcomp)6.) * wfnc[sx][sy][sz]) / (A * A);

          t_kin[sx][sy][sz] = c_sq * laplacian;
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