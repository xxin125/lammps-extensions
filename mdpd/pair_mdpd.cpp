// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Xinxin Deng (TU Darmstadt)
------------------------------------------------------------------------- */

#include "pair_mdpd.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "random_mars.h"
#include "update.h"
#include "compute.h"
#include "math_const.h"

#include <cmath>

using namespace LAMMPS_NS;
using MathConst::MY_PI;

#define EPSILON 1.0e-10


#define LCGA 0x4beb5d59    // Full period 32 bit LCG
#define LCGC 0x2600e1f7
#define oWeylPeriod 0xda879add    // Prime period 3666320093
#define oWeylOffset 0x8009d14b
#define TWO_N32 0.232830643653869628906250e-9f /* 2^-32 */
// specifically implemented for steps = 1; high = 1.0; low = -1.0
// returns uniformly distributed random numbers u in [-1.0;1.0] using TEA8
// then multiply u with sqrt(3) to "match" with a normal random distribution
// Afshar et al. mutlplies u in [-0.5;0.5] with sqrt(12)

#define numtyp double
#define SQRT3 (numtyp) 1.7320508075688772935274463
#define k0 0xA341316C
#define k1 0xC8013EA4
#define k2 0xAD90777D
#define k3 0x7E95761E
#define delta 0x9e3779b9
#define rounds 8
#define saru(seed1, seed2, seed, timestep, randnum)                           \
  {                                                                           \
    unsigned int seed3 = seed + timestep;                                     \
    seed3 ^= (seed1 << 7) ^ (seed2 >> 6);                                     \
    seed2 += (seed1 >> 4) ^ (seed3 >> 15);                                    \
    seed1 ^= (seed2 << 9) + (seed3 << 8);                                     \
    seed3 ^= 0xA5366B4D * ((seed2 >> 11) ^ (seed1 << 1));                     \
    seed2 += 0x72BE1579 * ((seed1 << 4) ^ (seed3 >> 16));                     \
    seed1 ^= 0x3F38A6ED * ((seed3 >> 5) ^ (((signed int) seed2) >> 22));      \
    seed2 += seed1 * seed3;                                                   \
    seed1 += seed3 ^ (seed2 >> 2);                                            \
    seed2 ^= ((signed int) seed2) >> 17;                                      \
    unsigned int state = 0x79dedea3 * (seed1 ^ (((signed int) seed1) >> 14)); \
    unsigned int wstate = (state + seed2) ^ (((signed int) state) >> 8);      \
    state = state + (wstate * (wstate ^ 0xdddf97f5));                         \
    wstate = 0xABCB96F7 + (wstate >> 1);                                      \
    unsigned int sum = 0;                                                     \
    for (int i = 0; i < rounds; i++) {                                        \
      sum += delta;                                                           \
      state += ((wstate << 4) + k0) ^ (wstate + sum) ^ ((wstate >> 5) + k1);  \
      wstate += ((state << 4) + k2) ^ (state + sum) ^ ((state >> 5) + k3);    \
    }                                                                         \
    unsigned int v = (state ^ (state >> 26)) + wstate;                        \
    unsigned int s = (signed int) ((v ^ (v >> 20)) * 0x6957f5a7);             \
    randnum = SQRT3 * (s * TWO_N32 * (numtyp) 2.0 - (numtyp) 1.0);            \
  }



/* ---------------------------------------------------------------------- */

PairMDPD::PairMDPD(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
  nmax = 0;
  rho = nullptr;
  comm_reverse = 1;
  comm_forward = 1;
  single_enable = 0;
  manybody_flag = 1;
}

/* ---------------------------------------------------------------------- */

PairMDPD::~PairMDPD()
{
  if (copymode) return;

  memory->destroy(rho);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(a0);
    memory->destroy(b0);
    memory->destroy(gamma);
    memory->destroy(sigma);
    memory->destroy(rc);
    memory->destroy(rd);
  }
}

/* ---------------------------------------------------------------------- */

void PairMDPD::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double vxtmp,vytmp,vztmp,delvx,delvy,delvz;
  double rsq,r,rinv,dot,w_c,randnum,factor_dpd;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double w_d,rdij,pref,rhoi,rhoj;
  tagint itag, jtag;
  double u_c, mb_factor, u_d_i, u_d_j;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  // from pair_eam.cpp

  if (atom->nmax > nmax) {
    memory->destroy(rho);
    nmax = atom->nmax;
    memory->create(rho,nmax,"pair:rho");
  }

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double dtinvsqrt = 1.0/sqrt(update->dt);
  int timestep = (int) update->ntimestep;


  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // zero out density

  if (newton_pair) {
    for (i = 0; i < nall; i++) rho[i] = 0.0;
  } else for (i = 0; i < nlocal; i++) rho[i] = 0.0;

  // rho = density at each atom
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      r = sqrt(rsq);
      jtype = type[j];
      rdij = rd[itype][jtype];

      if (r < rdij) {
        w_d = 1.0 - r/rdij;
        pref = 15.0/(2.0*MY_PI*rdij*rdij*rdij);
        rho[i] += pref*w_d*w_d;
        if (newton_pair || j < nlocal) {
          rho[j] += pref*w_d*w_d;
        }
      }
    }
  }

  // communicate and sum densities

  if (newton_pair) comm->reverse_comm(this);
  comm->forward_comm(this);

  // compute forces on each atom
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    itag = tag[i];
    rhoi = rho[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_dpd = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];
      jtag = tag[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        if (r < EPSILON) continue;     // r can be 0.0 in DPD systems
        rinv = 1.0/r;
        delvx = vxtmp - v[j][0];
        delvy = vytmp - v[j][1];
        delvz = vztmp - v[j][2];
        dot = delx*delvx + dely*delvy + delz*delvz;

        w_c = 1.0 - r/rc[itype][jtype];
        w_d = 1.0 - r/rd[itype][jtype];
        w_d = MAX(w_d,0.0);

        rhoj = rho[j];

        unsigned int tag1 = itag, tag2 = jtag;
        if (tag1 > tag2) {
          tag1 = jtag;
          tag2 = itag;
        }

        randnum = 0.0;
        saru(tag1, tag2, seed, timestep, randnum);


        // conservative force = a0 * w_c + b0 * (rhoi + rhoj) * w_d
        // drag force = -gamma * w_c^2 * (delx dot delv) / r
        // random force = sigma * w_c * rnd * dtinvsqrt;

        fpair = a0[itype][jtype]*w_c + b0[itype][jtype]*(rhoi+rhoj)*w_d;
        fpair -= gamma[itype][jtype]*w_c*w_c*dot*rinv;
        fpair += sigma[itype][jtype]*w_c*randnum*dtinvsqrt;
        fpair *= factor_dpd*rinv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          u_c = 0.5*a0[itype][jtype]*rc[itype][jtype]*w_c*w_c;
          mb_factor = 0.25*b0[itype][jtype]*rd[itype][jtype]*w_d*w_d;
          u_d_i = mb_factor*rhoi; 
          u_d_j = mb_factor*rhoj;
          evdwl = u_c+u_d_i+u_d_j;

          evdwl *= factor_dpd;
          u_c *= factor_dpd;
          u_d_i *= factor_dpd;
          u_d_j *= factor_dpd;
        }

        if (evflag) ev_tally_mdpd(i,j,nlocal,newton_pair,
                             evdwl,u_c,u_d_i,u_d_j,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   ev_tally_mdpd 
------------------------------------------------------------------------- */

void PairMDPD::ev_tally_mdpd(int i, int j, int nlocal, int newton_pair,
                    double evdwl, double u_c, double u_d_i, double u_d_j, double fpair,
                    double delx, double dely, double delz)
{
  double evdwlhalf,ecoulhalf,epairhalf,v[6];

  if (eflag_either) {
    if (eflag_global) {
      if (newton_pair) {
        eng_vdwl += evdwl;
      } else {
        evdwlhalf = 0.5*evdwl;
        if (i < nlocal) {
          eng_vdwl += evdwlhalf;
        }
        if (j < nlocal) {
          eng_vdwl += evdwlhalf;
        }
      }
    }
    if (eflag_atom) {
      if (newton_pair || i < nlocal) eatom[i] += 0.5*u_c+u_d_i;
      if (newton_pair || j < nlocal) eatom[j] += 0.5*u_c+u_d_j;
    }
  }

  if (vflag_either) {
    v[0] = delx*delx*fpair;
    v[1] = dely*dely*fpair;
    v[2] = delz*delz*fpair;
    v[3] = delx*dely*fpair;
    v[4] = delx*delz*fpair;
    v[5] = dely*delz*fpair;

    if (vflag_global) {
      if (newton_pair) {
        virial[0] += v[0];
        virial[1] += v[1];
        virial[2] += v[2];
        virial[3] += v[3];
        virial[4] += v[4];
        virial[5] += v[5];
      } else {
        if (i < nlocal) {
          virial[0] += 0.5*v[0];
          virial[1] += 0.5*v[1];
          virial[2] += 0.5*v[2];
          virial[3] += 0.5*v[3];
          virial[4] += 0.5*v[4];
          virial[5] += 0.5*v[5];
        }
        if (j < nlocal) {
          virial[0] += 0.5*v[0];
          virial[1] += 0.5*v[1];
          virial[2] += 0.5*v[2];
          virial[3] += 0.5*v[3];
          virial[4] += 0.5*v[4];
          virial[5] += 0.5*v[5];
        }
      }
    }

    if (vflag_atom) {
      if (newton_pair || i < nlocal) {
        vatom[i][0] += 0.5*v[0];
        vatom[i][1] += 0.5*v[1];
        vatom[i][2] += 0.5*v[2];
        vatom[i][3] += 0.5*v[3];
        vatom[i][4] += 0.5*v[4];
        vatom[i][5] += 0.5*v[5];
      }
      if (newton_pair || j < nlocal) {
        vatom[j][0] += 0.5*v[0];
        vatom[j][1] += 0.5*v[1];
        vatom[j][2] += 0.5*v[2];
        vatom[j][3] += 0.5*v[3];
        vatom[j][4] += 0.5*v[4];
        vatom[j][5] += 0.5*v[5];
      }
    }
  }

  if (num_tally_compute > 0) {
    did_tally_flag = 1;
    for (int k=0; k < num_tally_compute; ++k) {
      Compute *c = list_tally_compute[k];
      c->pair_tally_callback(i, j, nlocal, newton_pair,
                             evdwl, 0.0, fpair, delx, dely, delz);
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairMDPD::allocate()
{
  int i,j;
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(a0,n+1,n+1,"pair:a0");
  memory->create(b0,n+1,n+1,"pair:b0");
  memory->create(rc,n+1,n+1,"pair:rc");
  memory->create(rd,n+1,n+1,"pair:rd");
  memory->create(gamma,n+1,n+1,"pair:gamma");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  for (i = 0; i <= atom->ntypes; i++)
    for (j = 0; j <= atom->ntypes; j++)
      sigma[i][j] = gamma[i][j] = 0.0;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMDPD::settings(int narg, char **arg)
{
  if (narg != 4) error->all(FLERR,"Illegal pair_style command");

  temperature = utils::numeric(FLERR,arg[0],false,lmp);
  cut_global = utils::numeric(FLERR,arg[1],false,lmp);
  rd_global = utils::numeric(FLERR,arg[2],false,lmp);
  seed = utils::inumeric(FLERR,arg[3],false,lmp);

  if (cut_global < rd_global) error->all(FLERR,"Incorrect args for pair_style\n cutA should be larger than cutB.");

  // initialize Marsaglia RNG with processor-unique seed

  if (seed <= 0) error->all(FLERR,"Illegal pair_style command");

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) rc[i][j] = cut_global;
        if (setflag[i][j]) rd[i][j] = rd_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMDPD::coeff(int narg, char **arg)
{
  if (narg != 5 && narg != 7)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double a0_one = utils::numeric(FLERR,arg[2],false,lmp);
  double b0_one = utils::numeric(FLERR,arg[3],false,lmp);
  double gamma_one = utils::numeric(FLERR,arg[4],false,lmp);

  double cut_one = cut_global;
  double rd_one = rd_global;
  if (narg == 7) {
    cut_one = utils::numeric(FLERR,arg[5],false,lmp);
    rd_one = utils::numeric(FLERR,arg[6],false,lmp);
    if (cut_one < rd_one) error->all(FLERR,"Incorrect args for pair coefficients\n cutA should be larger than cutB.");
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      a0[i][j] = a0_one;
      b0[i][j] = b0_one;
      gamma[i][j] = gamma_one;
      rc[i][j] = cut_one;
      rd[i][j] = rd_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairMDPD::init_style()
{
  if (comm->ghost_velocity == 0)
    error->all(FLERR,"Pair mdpd requires ghost atoms store velocity");

  // if newton off, forces between atoms ij will be double computed
  // using different random numbers

  if (force->newton_pair == 0 && comm->me == 0)
    error->warning(FLERR, "Pair mdpd needs newton pair on for momentum conservation");

  neighbor->add_request(this);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMDPD::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  sigma[i][j] = sqrt(2.0*force->boltz*temperature*gamma[i][j]);

  a0[j][i] = a0[i][j];
  b0[j][i] = b0[i][j];
  gamma[j][i] = gamma[i][j];
  sigma[j][i] = sigma[i][j];
  rc[j][i] = rc[i][j];
  rd[j][i] = rd[i][j];

  return rc[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMDPD::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&a0[i][j],sizeof(double),1,fp);
        fwrite(&b0[i][j],sizeof(double),1,fp);
        fwrite(&gamma[i][j],sizeof(double),1,fp);
        fwrite(&rc[i][j],sizeof(double),1,fp);
        fwrite(&rd[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMDPD::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&a0[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&b0[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&gamma[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&rc[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&rd[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&a0[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&b0[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&gamma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&rc[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&rd[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMDPD::write_restart_settings(FILE *fp)
{
  fwrite(&temperature,sizeof(double),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&rd_global,sizeof(double),1,fp);
  fwrite(&seed,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMDPD::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&temperature,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&cut_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&rd_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&seed,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&temperature,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&rd_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&seed,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);

  // initialize Marsaglia RNG with processor-unique seed
  // same seed that pair_style command initially specified
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairMDPD::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g\n",i,a0[i][i],b0[i][i],gamma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairMDPD::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g\n",i,j,a0[i][j],b0[i][j],gamma[i][j],rc[i][j],rd[i][j]);
}


/* ---------------------------------------------------------------------- */

int PairMDPD::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) buf[m++] = rho[i];
  return m;
}

/* ---------------------------------------------------------------------- */

void PairMDPD::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    rho[j] += buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int PairMDPD::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = rho[j];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairMDPD::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) rho[i] = buf[m++];
}