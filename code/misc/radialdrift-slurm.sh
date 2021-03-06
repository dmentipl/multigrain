#!/bin/bash
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=radialdrift
#SBATCH --output=radialdrift01.swmout
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=daniel.mentiplay@monash.edu
#SBATCH --time=7-00:00:00
#SBATCH --mem=16G

INFILE='radialdrift.in'

echo "HOSTNAME = $HOSTNAME"
echo "HOSTTYPE = $HOSTTYPE"
echo Time is "$(date)"
echo Directory is "$(pwd)"

ulimit -s unlimited
export OMP_SCHEDULE="dynamic"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_STACKSIZE=2048m

module load iccifort/2016.2.181-gcc-6.4.0
module load openmpi/3.0.0
module load hdf5/1.10.1

module list

echo "starting phantom run..."
OUTFILE=$(grep logfile $INFILE | tr -s ' ' | cut -d' ' -f4)
echo "writing output to $OUTFILE"
./phantom $INFILE >& "$OUTFILE"
