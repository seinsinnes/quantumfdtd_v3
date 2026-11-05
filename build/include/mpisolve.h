/*

   mpisolve.h

   Copyright (c) Michael Strickland

   GNU General Public License (GPLv3)
   See detailed text in license directory

*/

#ifndef __mpisolve_h__
#define __mpisolve_h__

typedef complex<double> dcomp;

// Existing quantumfdtd_v3 externs...
extern int NUMX, NUM, STEPS, UPDATE, SNAPUPDATE;
extern double A, EPS, SIG, MASS, SIGMA, TOLERANCE;

extern int POTENTIAL, INITCONDTYPE, INITCONDAXIS, INITSYMMETRY, NF, SAVEWAVEFNCS, DUMPSNAPS, SNAPDUMP;
extern double TC, SIGMA, XI, eB, Kx, SPINEXP;

// --- Acoustic Helmholtz Renderer Parameters ---
extern double C_MEDIUM;      // Speed of sound in surrounding fluid/air (m/s)
extern double C_OBJECT;      // Speed of sound inside the VDB geometry (m/s)
extern double FREQ;          // Transducer frequency in Hz (if <= 0, computed from WAVELENGTH)
extern double WAVELENGTH;    // Wavelength in grid voxels (used when FREQ <= 0)
extern double SOURCE_AMP;    // Transducer driving amplitude
extern int    BEAM_RADIUS;   // Transducer disc radius in voxels
extern int    SPONGE;        // Absorbing boundary layer thickness in voxels
extern double SPONGE_DAMP;   // Absorbing boundary exponential damping factor
extern char  *VDBFILE;       // Path to input geometry VDB file

extern int KINTERM;               //R: ADDED RELATIVISTIC GLOBAL VARIABLE: 20/03/2018
extern int SAVEPOT;               //R: ADDED GLOBAL VARIABLE
extern int SAVEDECAY;             //R: ADDED GLOBAL VARIABLE
extern double POTCRITR, POTFLATR; //R: ADDED GLOBAL VARIABLE

extern char *EXTPOT;   //R: ADDED EXTERNAL FILES POTENTIAL AND POTENTIAL SUBSTRACTION
extern char *DATAFOLD; //R: ADDED EXTERNAL DIRECTORY FOR OUTPUT DATA

extern int    AUTO_ALIGN;    // 1 = Auto-fit VDB bbox to grid, 0 = Manual [0, NUM*A] coords
extern double OBJECT_FILL;   // Fraction of grid spanned by object's longest axis (0.1 to 0.9)

extern int nodeID, numNodes, debug;

extern dcomp energyCollect;
extern dcomp normalizationCollect;
extern dcomp vInfinityCollect;
extern dcomp rRMS2Collect;
extern dcomp xAvgCollect;
extern dcomp yAvgCollect;
extern dcomp zAvgCollect;
extern dcomp ***t_kin;

/* debug values */
#define DEBUG_OFF 0
#define DEBUG_ON 1
#define DEBUG_FULL 2

/* message tags */
#define HELLO 1
#define DONE 2
#define SYNC_LEFT 3
#define SYNC_LEFT_MESSAGE 4
#define SYNC_RIGHT 5
#define SYNC_RIGHT_MESSAGE 6

// the main solve routines
void solveInitialize();
void solve();
void solveFinalize();
void evolve(int);
void findExcitedStates(double, int);

// boundary sync stuff
void syncBoundaries(dcomp ***wfnc);
void sendLeftBoundary(dcomp ***wfnc);
void sendRightBoundary(dcomp ***wfnc);
void receiveLeftBoundary();
void receiveRightBoundary();
void loadRightBoundaryFromBuffer(dcomp ***wfnc);
void loadLeftBoundaryFromBuffer(dcomp ***wfnc);

// fftw3 options
void wfncKinetic(dcomp ***wfnc);

#endif /* __mpisolve_h__ */
