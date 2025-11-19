#!/bin/bash
# Simple script to run Ga liquid simulation with LAMMPS (CPU version)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Liquid Gallium LAMMPS Simulation"
echo "  CPU-based EAM potential"
echo "=========================================="
echo ""

# Check if LAMMPS is installed
if command -v lammps &> /dev/null; then
    LAMMPS_CMD="lammps"
elif command -v lmp &> /dev/null; then
    LAMMPS_CMD="lmp"
elif command -v lmp_mpi &> /dev/null; then
    LAMMPS_CMD="lmp_mpi"
else
    echo -e "${RED}Error: LAMMPS not found!${NC}"
    echo "Please install LAMMPS first."
    echo "  Ubuntu: sudo apt install lammps"
    echo "  Conda:  conda install -c conda-forge lammps"
    exit 1
fi

echo -e "${GREEN}✓ LAMMPS found: $LAMMPS_CMD${NC}"

# Check if EAM potential file exists
if [ ! -f "Ga_belashchenko2012.eam.alloy" ]; then
    echo -e "${RED}Error: EAM potential file not found!${NC}"
    echo "Please make sure Ga_belashchenko2012.eam.alloy is in the current directory."
    exit 1
fi

echo -e "${GREEN}✓ EAM potential file found${NC}"

# Check if input file exists
if [ ! -f "in.ga_liquid" ]; then
    echo -e "${RED}Error: Input file not found!${NC}"
    echo "Please make sure in.ga_liquid is in the current directory."
    exit 1
fi

echo -e "${GREEN}✓ Input file found${NC}"
echo ""

# Ask for number of cores
read -p "Enter number of CPU cores to use (default: 4): " NCORES
NCORES=${NCORES:-4}

echo ""
echo "=========================================="
echo "  Simulation Settings"
echo "=========================================="
echo "  Cores:        $NCORES"
echo "  Temperature:  293 K"
echo "  System:       Liquid Gallium"
echo "  Steps:        ~85,000 (total)"
echo "  Time:         ~85 ps"
echo "=========================================="
echo ""

# Ask for confirmation
read -p "Start simulation? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Simulation cancelled."
    exit 0
fi

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output_ga_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo -e "${YELLOW}Starting simulation...${NC}"
echo "Output will be saved to: $OUTPUT_DIR"
echo ""

# Run simulation
START_TIME=$(date +%s)

if [ $NCORES -eq 1 ]; then
    # Serial execution
    $LAMMPS_CMD -in in.ga_liquid > "$OUTPUT_DIR/lammps.log" 2>&1
else
    # Parallel execution with MPI
    mpirun -np $NCORES $LAMMPS_CMD -in in.ga_liquid > "$OUTPUT_DIR/lammps.log" 2>&1
fi

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Move output files to output directory
mv -f log.lammps "$OUTPUT_DIR/" 2>/dev/null
mv -f dump.ga.lammpstrj "$OUTPUT_DIR/" 2>/dev/null
mv -f rdf.ga.dat "$OUTPUT_DIR/" 2>/dev/null
mv -f msd.ga.dat "$OUTPUT_DIR/" 2>/dev/null
mv -f final.ga.data "$OUTPUT_DIR/" 2>/dev/null
mv -f restart.ga "$OUTPUT_DIR/" 2>/dev/null

echo ""
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Simulation completed successfully!${NC}"
    echo "=========================================="
    echo "  Elapsed time: ${ELAPSED} seconds ($((ELAPSED/60)) min $((ELAPSED%60)) sec)"
    echo ""
    echo "Output files in: $OUTPUT_DIR/"
    echo "  - lammps.log          Main output log"
    echo "  - log.lammps          Thermo data"
    echo "  - dump.ga.lammpstrj   Trajectory"
    echo "  - rdf.ga.dat          Radial distribution function"
    echo "  - msd.ga.dat          Mean square displacement"
    echo "  - final.ga.data       Final structure"
    echo "  - restart.ga          Restart file"
    echo ""
    
    # Extract and display final properties
    if [ -f "$OUTPUT_DIR/lammps.log" ]; then
        echo "Final properties:"
        grep "Final density:" "$OUTPUT_DIR/lammps.log" 2>/dev/null
        grep "Average temperature:" "$OUTPUT_DIR/lammps.log" 2>/dev/null
        grep "Average pressure:" "$OUTPUT_DIR/lammps.log" 2>/dev/null
    fi
    
    echo ""
    echo "Visualization:"
    echo "  OVITO:    ovito $OUTPUT_DIR/dump.ga.lammpstrj"
    echo "  VMD:      vmd $OUTPUT_DIR/dump.ga.lammpstrj"
    echo ""
    echo "Analysis:"
    echo "  RDF plot: python plot_rdf.py $OUTPUT_DIR/rdf.ga.dat"
    echo "  MSD plot: python plot_msd.py $OUTPUT_DIR/msd.ga.dat"
    
else
    echo -e "${RED}✗ Simulation failed!${NC}"
    echo "=========================================="
    echo "  Check log file for errors: $OUTPUT_DIR/lammps.log"
    echo ""
    echo "Common issues:"
    echo "  1. Missing MANYBODY package in LAMMPS"
    echo "  2. Incorrect EAM file path"
    echo "  3. Insufficient memory"
    echo "  4. Unstable timestep (try reducing)"
fi

echo "=========================================="
echo ""