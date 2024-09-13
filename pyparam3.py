#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:43:43 2019

@author: Åsmund Røhr Kjendseth (and some stolen pices of code from here and there)

IMPORTANT INFORMATION:
    
    Set the PATHS to ORCA and AMBER programs in the script below.
    
    Use the following input for ORCA (content in brackes is up to you). Use "orca" as basename.
        
        ! [UKS B3LYP  OPT def2-SVP D4 SlowConv TightSCF] keepdens  AnFreq
        

"""

orca_path = "/Users/asmunroh/Prog/orca_5_0_4_macosx_arm64_openmpi411/"
antechamber_path = "/Users/asmunroh/opt/anaconda3/envs/AmberTools24/bin/"


# "/Users/asmunroh/Prog/orca_5_0_4_macosx_arm64_openmpi411/"
# "/Users/asmunroh/Prog/orca_6_0_0/"


import sys
import subprocess
import scipy as linalg
import numpy as np
import math, random
import time as time
from math import cos, sin, pi, sqrt
import argparse
import pandas as pd
import re


parser = argparse.ArgumentParser(description='RESP calculation from ORCA output')

parser.add_argument(
    '-name',
    default="MOL",
    metavar='',
    help='3 letter  name for parameterized residue. E.g. TYF, CUB, FEO, etc (default: MOL)'
)

parser.add_argument(
    '-resp',
    default="on",
    metavar='',
    help='Turn RESP calculation from ORCA densities on/off (default: on)'
)

parser.add_argument(
    '-force',
    default="on",
    metavar='',
    help='Turn on Seminaro ff generation (default: on)'
)

parser.add_argument(
    '-p',
    default=150,
    metavar='',
    help='provide an number of ESP points (default: 150, mimic resp protocol with Gaussian)'
)

parser.add_argument(
    '-l',
    default=10,
    metavar='',
    help='provide an number of ESP layers (default: 10, mimic resp protocol with Gaussian)'
)


parser.add_argument(
    '-at',
    default="None",
    metavar='',
    help='Provide atom name of special interest in frcmod file (default: none, e.g. CT, ). Not updated function.'
)

parser.add_argument(
    '-respfile',
    default="None",
    metavar='',
    help='Provide atom name of special interest in frcmod file (default: None, e.g. CT, ). Not updated function.'
)


my_namespace = parser.parse_args()


# Function to find system charge from orca input file
def find_charge(orca_input_file):
    with open(orca_input_file, 'r') as file:
        data = file.read().splitlines()
        
        for line in data:
            # Match patterns like *xyz, * xyz, *pdbfile, etc.
            match = re.search(r'^\*\s*(\w+)\s+(-?\d+)\s+(-?\d+)', line)
            if match:
                charge = int(match.group(2))  # The second matched group is the charge
                return charge
        raise ValueError("Charge not found in the input file")

# Function to find system multiplicity from orca input file
def find_mult(orca_input_file):
    with open(orca_input_file, 'r') as file:
        data = file.read().splitlines()
        
        for line in data:
            # Match patterns like *xyz, * xyz, *pdbfile, etc.
            match = re.search(r'^\*\s*(\w+)\s+(-?\d+)\s+(-?\d+)', line)
            if match:
                mult = int(match.group(3))  # The third matched group is the multiplicity
                return mult
        raise ValueError("Multiplicity not found in the input file")


# Function to generate a unit sphere
def fibonacci_sphere(samples=1,randomize=True):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])
    return points

# Alternative function to generate a unit sphere
def sunflower_sphere(samples=1):
    """ each point you get will be of form 'x, y, z'; in cartesian coordinates
        eg. the 'l2 distance' from the origion [0., 0., 0.] for each point will be 1.0 
        ------------
        converted from:  http://web.archive.org/web/20120421191837/http://www.cgafaq.info/wiki/Evenly_distributed_points_on_sphere ) 
    """
    dlong = pi*(3.0-sqrt(5.0))  # ~2.39996323 
    dz   =  2.0/samples
    long =  0.0
    z    =  1.0 - dz/2.0
    points =[]
    for k in range( 0, samples): 
        r    = sqrt(1.0-z*z)
        ptNew = (cos(long)*r, sin(long)*r, z)
        points.append( ptNew )
        z    = z - dz
        long = long + dlong
    return points

# Function that generates input file for orca_vpot.   sphere_scale set to 0.1
def gen_esp_points(x, y, z, npoints):
    mep_tmp = np.zeros((1,3))
    all_shells = np.zeros((1,3))

    for h in range(0, len(shell_list)):
        molecule_shell = np.zeros((1,3))
        for i in range(0, len(x)):
            shell = np.array(sunflower_sphere(samples= int(npoints * (shell_list[h] ** 2 / shell_list[0] ** 2)))) * shell_list[h] * 0.01 * ang_to_au * float(at_radii[elements.index(atoms[i])])
            single_shells = np.empty((len(shell),3))
            for k in range(0, len(shell)):
                vec = np.array([shell[k,0] + x[i] * ang_to_au, shell[k,1] + y[i] * ang_to_au, shell[k,2] + z[i] * ang_to_au])
                mep_tmp = np.vstack((mep_tmp, vec))
                
                single_shells[k,0] = shell[k,0] + x[i] * ang_to_au
                single_shells[k,1] = shell[k,1] + y[i] * ang_to_au
                single_shells[k,2] = shell[k,2] + z[i] * ang_to_au
        
            molecule_shell = np.vstack((molecule_shell, single_shells))      
        
        # Delete points due to overlapping spheres
        delete_index = []
        for i in range(0, len(atoms)):
            for j in range(0, len(molecule_shell)):
                if (np.linalg.norm(np.array([x[i] * ang_to_au - molecule_shell[j,0], y[i] * ang_to_au - molecule_shell[j,1], z[i] * ang_to_au - molecule_shell[j,2]])) < (0.999 * shell_list[h] * 0.01 * ang_to_au * float(at_radii[elements.index(atoms[i])]))):
                    delete_index.append(j)
        molecule_shell = np.delete(molecule_shell, delete_index, axis=0)
        molecule_shell = np.delete(molecule_shell, 0, axis=0)
        all_shells = np.vstack((all_shells, molecule_shell, ))
    
    all_shells = np.delete(all_shells, 0, axis=0)

    #Print xyz file of all points used to calculate ESP
    np.savetxt("ESP_points_for_visualization.xyz", np.hstack((np.full((len(all_shells), 1),"X"), all_shells)), fmt='%s')

    mep_inp = open(basename + "_mep.inp", "w")
    mep_inp.write("{0:d}\n".format(len(all_shells))) # Change nponits here
    for i in range(0, len(all_shells)):
        mep_inp.write("{0:12.6f} {1:12.6f} {2:12.6f}\n".format(all_shells[i,0], all_shells[i,1], all_shells[i,2]))
    mep_inp.close()
    return

# Function that read the output from orca_vpot
def read_vpot(vpot):
    vx = []
    vy = []
    vz = []
    v = []
    f = open(vpot, "r")
    f.readline()
    for line in f:
        data = line.split()
        vx.append(float(data[0]))
        vy.append(float(data[1]))
        vz.append(float(data[2]))
        v.append(float(data[3]))
    f.close()
    return np.array(vx), np.array(vy), np.array(vz), np.array(v)

# Function to read cartesian coordinates from a orca.xyz file. The two first lines are n-atoms and comment.
def read_xyz(xyz):
    atoms = []
    x = []
    y = []
    z = []
    f = open(xyz, "r")
    f.readline()
    f.readline()
    for line in f:
        data = line.split()
        atoms.append(data[0])
        x.append(float(data[1]))
        y.append(float(data[2]))
        z.append(float(data[3]))
    f.close()
    xyz = np.column_stack((x,y,z))
    return atoms, np.array(x), np.array(y), np.array(z), xyz

# Function to read Hessian matrix from the file orca.hess
def read_hessian(natoms):    
    skiplines = 0
    with open('orca.hess', 'r') as file:
        for i, line in enumerate(file, 1):  # Start counting from line 1
            if "$hessian" in line:
                skiplines = i+1
                break
    
    n_chunks = int(np.ceil(3 * natoms / 5))
    df_hess = pd.read_csv('orca.hess', sep="\\s+", skiprows=skiplines, nrows=(3 * natoms * n_chunks + n_chunks) - 1)
    arr = df_hess.values
    N = df_hess.index.max()+1
    arr = np.delete(arr, np.arange(N, len(arr), N+1), axis=0)
    chunks = np.split(arr, np.arange(N, len(arr), N))
    Hessian = pd.DataFrame(np.hstack(chunks)).dropna(axis=1)  # Units are Hartree/bohr**2
    Hessian = (627.509469/0.52917721092 ** 2) * Hessian       # Hessian units are now kcal/mol*Å**2  
    Hessian = np.array(Hessian)
    return Hessian


    
# Function that determine connectivity between defined atoms (see if tests below) in the orca.xyz file 
def get_connectivity(atoms, array_x, array_y, array_z):
    connectivity = ['E1', 'Nr.','E2', 'Nr.', 'Bond length']
    for i in range(0, len(atoms)):
        for j in range(0, len(atoms)):
            bond_length = np.linalg.norm(np.array([np.array(x)[i], np.array(y)[i], np.array(z)[i]]) - np.array([np.array(x)[j], np.array(y)[j], np.array(z)[j]]))
            if (atoms[i] == 'C' and atoms[j] == 'H' and 0 < bond_length < 1.5 ):
                connectivity.extend([atoms[i], i, atoms[j], j, bond_length])
         
            if (atoms[i] == 'N' and atoms[j] == 'H' and 0 < bond_length < 1.5 ):
                connectivity.extend([atoms[i], i, atoms[j], j, bond_length])
           
            if (atoms[i] == 'O' and atoms[j] == 'H' and 0 < bond_length < 1.5 ):
                connectivity.extend([atoms[i], i, atoms[j], j, bond_length])
                
            if (atoms[i] == 'O' and atoms[j] == 'O' and 0 < bond_length < 1.7 ):
                connectivity.extend([atoms[i], i, atoms[j], j, bond_length])

    
            if (atoms[i] == 'C' and atoms[j] == 'C' and 0 < bond_length < 2.0 ):
                connectivity.extend([atoms[i], i, atoms[j], j, bond_length])
           
            if (atoms[i] == 'C' and atoms[j] == 'N' and 0 < bond_length < 2.0 ):
                connectivity.extend([atoms[i], i, atoms[j], j, bond_length])

            if (atoms[i] == 'C' and atoms[j] == 'O' and 0 < bond_length < 2.0 ):
                connectivity.extend([atoms[i], i, atoms[j], j, bond_length])

            if (atoms[i] == 'C' and atoms[j] == 'S' and 0 < bond_length < 2.0 ):
                connectivity.extend([atoms[i], i, atoms[j], j, bond_length])

            if (atoms[i] == 'S' and atoms[j] == 'S' and 0 < bond_length < 2.4 ):
                connectivity.extend([atoms[i], i, atoms[j], j, bond_length])

            if (atoms[i] == 'N' and atoms[j] == 'Cu' and 0 < bond_length < 2.5 ):
                connectivity.extend([atoms[i], i, atoms[j], j, bond_length])
            
            if (atoms[i] == 'O' and atoms[j] == 'Cu' and 0 < bond_length < 3.0 ):
                connectivity.extend([atoms[i], i, atoms[j], j, bond_length])
            
            if (atoms[i] == 'N' and atoms[j] == 'Fe' and 0 < bond_length < 2.5 ):
                connectivity.extend([atoms[i], i, atoms[j], j, bond_length])
            
            if (atoms[i] == 'O' and atoms[j] == 'Fe' and 0 < bond_length < 3.0 ):
                connectivity.extend([atoms[i], i, atoms[j], j, bond_length])

    connectivity_table = np.array(connectivity).reshape(int(len(connectivity) / 5), 5)
    for i in range(0, len(connectivity_table)):
        if (connectivity_table[i,1] > connectivity_table[i,3]):
            a = connectivity_table[i,1]
            b = connectivity_table[i,3]
            c = connectivity_table[i,0]
            d = connectivity_table[i,2]
            connectivity_table[i,1] = b
            connectivity_table[i,3] = a
            connectivity_table[i,0] = d
            connectivity_table[i,2] = c
    connectivity_table_header = connectivity_table[0,:]
    connectivity_table = np.delete(connectivity_table, 0, axis=0)
    connectivity_table = connectivity_table[np.lexsort((connectivity_table[:, 1], connectivity_table[:,3]))]
    
    delete_index = []
    for i in range(0, len(connectivity_table) - 1):
            if (((connectivity_table[i,1] == connectivity_table[i + 1,1]) and (connectivity_table[i, 3]) == (connectivity_table[i + 1,3] ))):
               delete_index.append(i)
    connectivity_table = np.delete(connectivity_table, delete_index, axis=0)
    connectivity_index = np.int64(connectivity_table[0:len(connectivity_table),[1,3]])
    connectivity_table = np.vstack((connectivity_table_header, connectivity_table))
    return connectivity_index, connectivity_table


# Find force field angles to be calculated from molecule connectivity
def find_angle_atoms(connectivity_index):
    angles = ['E1', 'Nr.','E2', 'Nr.', 'E3', 'Nr.','Angle']
    for i in range(0, len(connectivity_index)):
        for j in range(i, len(connectivity_index)):
            if (connectivity_index[i,0] == connectivity_index[j,0] and connectivity_index[i,1] != connectivity_index[j,1]):
                #print(connectivity_index[j,1], connectivity_index[i,0], connectivity_index[i,1])
                angles.extend([atoms[(connectivity_index[j,1])],(connectivity_index[j,1]), \
                atoms[(connectivity_index[i,0])], (connectivity_index[i,0]), \
                atoms[(connectivity_index[i,1])], (connectivity_index[i,1]),bond_angle(connectivity_index[j,1], connectivity_index[i,0], connectivity_index[i,1])])
            
            if (connectivity_index[i,1] == connectivity_index[j,0]) and (connectivity_index[i,0] != connectivity_index[j,1]):
                #print(connectivity_index[i,0], connectivity_index[i,1], connectivity_index[j,1])
                angles.extend([atoms[(connectivity_index[i,0])],(connectivity_index[i,0]), \
                atoms[(connectivity_index[i,1])], (connectivity_index[i,1]), \
                atoms[(connectivity_index[j,1])], (connectivity_index[j,1]),bond_angle(connectivity_index[i,0], connectivity_index[i,1], connectivity_index[j,1])])
    
            if (connectivity_index[i,1] == connectivity_index[j,1]) and (connectivity_index[i,0] != connectivity_index[j,0]):
                #print(connectivity_index[i,0], connectivity_index[i,1], connectivity_index[j,1])
                angles.extend([atoms[(connectivity_index[i,0])],(connectivity_index[i,0]), \
                atoms[(connectivity_index[i,1])], (connectivity_index[i,1]), \
                atoms[(connectivity_index[j,0])], (connectivity_index[j,0]),bond_angle(connectivity_index[i,0], connectivity_index[i,1], connectivity_index[j,0])])

    angles_table = np.array(angles).reshape(int(len(angles) / 7), 7)
    angles_index = np.array(np.delete(angles_table[:,[1, 3, 5]], 0, 0)).astype(int)
    return angles_table,angles_index
                
    

# Calculation of bond angle for atom i-j-k
def bond_angle(atom_i, atom_j, atom_k):
       v = xyz[atom_i,0:3] - xyz[atom_j,0:3]
       u = xyz[atom_k,0:3] - xyz[atom_j,0:3]
       angle = np.degrees(np.arccos(np.dot(v,u) / (np.linalg.norm(v) * np.linalg.norm(u))))
       return(angle)

# Calculation of unit vector for A-B
def unit_vector(xyz, atom_i, atom_j):
    u_XY = np.divide(xyz[atom_i,0:3] - xyz[atom_j,0:3], np.linalg.norm((xyz[atom_i,0:3] - xyz[atom_j,0:3])))
    return(u_XY)

# Calculation of unit vector N
def unit_vector_n(xyz, atom_i, atom_j, atom_k):
    n_ABC = np.cross(unit_vector(xyz, atom_k, atom_j),unit_vector(xyz, atom_i, atom_j)) / np.linalg.norm(np.cross(unit_vector(xyz, atom_k, atom_j),unit_vector(xyz, atom_i, atom_j)))
    u_PA = np.cross(n_ABC,unit_vector(xyz, atom_i, atom_j))
    u_PC = np.cross(unit_vector(xyz, atom_k, atom_j),n_ABC)
    return(u_PA, u_PC)

# Calculation of interaction matrix for two atoms, upper triagonal   
def interaction_matrix_up(Hessian, atom_i, atom_j):
    Int_matrix = -1 * Hessian[3* atom_i:3 * atom_i + 3,3 * atom_j:3 * atom_j + 3]
    return(Int_matrix)
    
# Calculation of interaction matrix for two atoms, lower triagonal   
def interaction_matrix_low(Hessian, atom_i, atom_j):
    Int_matrix = -1 * Hessian[3* atom_j:3 * atom_j + 3,3 * atom_i:3 * atom_i + 3]
    return(Int_matrix)

# Calculation of distance between two atoms      
def bond_length(atom_i, atom_j):
    bl = np.linalg.norm((xyz[atom_i,0:3] - xyz[atom_j,0:3]))
    return(bl)
   
# Calculation of bond force costant for  atoms i-j, upper triagonal
def calc_bond_fc_up(xyz, Hessian, atom_i, atom_j):
    # Calculation of unit vector for A-B
    #u_AB = unit_vector(xyz, atom_i, atom_j)
    # Calculation of eigenvalues and eigenvectors for A-B interaction
    E, V = np.linalg.eig(interaction_matrix_up(Hessian, atom_i, atom_j))
    fc_AB = (E[0] * abs(np.dot(unit_vector(xyz, atom_i, atom_j), V[:,0])) + E[1] * abs(np.dot(unit_vector(xyz, atom_i, atom_j), V[:,1])) + E[2] * abs(np.dot(unit_vector(xyz, atom_i, atom_j), V[:,2]))) / 2
    return(float(fc_AB.real))
    
# Calculation of bond force costant for  atoms i-j, lower triagonal
def calc_bond_fc_low(xyz, Hessian, atom_i, atom_j):
    # Calculation of unit vector for A-B
    #u_AB = unit_vector(xyz, atom_i, atom_j)
    # Calculation of eigenvalues and eigenvectors for A-B interaction
    E, V = np.linalg.eig(interaction_matrix_low(Hessian, atom_i, atom_j))
    fc_AB = (E[0] * abs(np.dot(unit_vector(xyz, atom_i, atom_j), V[:,0])) + E[1] * abs(np.dot(unit_vector(xyz, atom_i, atom_j), V[:,1])) + E[2] * abs(np.dot(unit_vector(xyz, atom_i, atom_j), V[:,2]))) / 2
    return(float(fc_AB.real))


# Calculation of angle force costant for atoms i-j-k
def calc_angle_fc(xyz, Hessian, atom_i, atom_j, atom_k):
    # Calculation of eigenvalues and eigenvectors for A-B interaction
    E_AB, V_AB = np.linalg.eig(interaction_matrix_up(Hessian, atom_i, atom_j))
    E_CB, V_CB = np.linalg.eig(interaction_matrix_up(Hessian, atom_k, atom_j))
    F_AB = bond_length(atom_i, atom_j) ** 2 * (E_AB[0] * abs(np.dot(unit_vector_n(xyz, atom_i, atom_j, atom_k)[0], V_AB[:,0])) + E_AB[1] * abs(np.dot(unit_vector_n(xyz, atom_i, atom_j, atom_k)[0], V_AB[:,1])) + E_AB[2] * abs(np.dot(unit_vector_n(xyz, atom_i, atom_j, atom_k)[0], V_AB[:,2])))
    F_CB = bond_length(atom_j, atom_k) ** 2 * (E_CB[0] * abs(np.dot(unit_vector_n(xyz, atom_i, atom_j, atom_k)[1], V_CB[:,0])) + E_CB[1] * abs(np.dot(unit_vector_n(xyz, atom_i, atom_j, atom_k)[1], V_CB[:,1])) + E_CB[2] * abs(np.dot(unit_vector_n(xyz, atom_i, atom_j, atom_k)[1], V_CB[:,2])))
    fc_ABC = (1/(1/F_AB + 1/F_CB))/2
    return(float(fc_ABC.real))

    

if __name__ == "__main__":

    
# Input variabels
    
    # Base for file name
    basename = "orca" #sys.argv[1]
    
    #Charge and multiplicity from orca.inp file
    charge = find_charge("orca.inp")
    mult   = find_mult("orca.inp")



    # Factor to convert from Å to Bohr
    ang_to_au = 1.0 / 0.5291772083

    # List of elements according to atomic number
    elements = [None,
         "H", "He",
         "Li", "Be",
         "B", "C", "N", "O", "F", "Ne",
         "Na", "Mg",
         "Al", "Si", "P", "S", "Cl", "Ar",
         "K", "Ca",
         "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
         "Ga", "Ge", "As", "Se", "Br", "Kr",
         "Rb", "Sr",
         "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
         "In", "Sn", "Sb", "Te", "I", "Xe",
         "Cs", "Ba",
         "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
         "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
         "Tl", "Pb", "Bi", "Po", "At", "Rn",
         "Fr", "Ra",
         "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No",
         "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Uub"]

    # List of VDW radii for elements. Transition metal radii should be checked and selected with care! Mertz Kollman radii for H+C+N+O+S+P
    at_radii = [None,
         "120", "140",
         "182", "Be",
         "B", "150", "150", "140", "147", "154",   # "old" values: "B", "170", "155", "152", "147", "154", replaced by info in gaussian log file.
         "227", "173",
         "Al", "210", "180", "175", "175", "188",
         "275", "Ca",
         "Sc", "Ti", "V", "Cr", "Mn", "150", "Co", "163", "180", "139",
         "Ga", "Ge", "As", "190", "185", "202",
         "Rb", "Sr",
         "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
         "In", "Sn", "Sb", "Te", "I", "Xe",
         "Cs", "Ba",
         "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
         "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
         "Tl", "Pb", "Bi", "Po", "At", "Rn",
         "Fr", "Ra",
         "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No",
         "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Uub"]  # Values in pm
    

# Program execution
    
    # Read atom names and xyz coordinates
    atoms, x, y, z, xyz = read_xyz(basename + ".xyz")
    natoms = len(atoms)
    
    # Set points on ESP surfaces (need more explanation)
    npoints = int(my_namespace.p)
    layers = int(my_namespace.l)
    
    # List of factors to multiply with the different atom VDW radii. First point to be found 1.4 x VDW radius from atom
    shell_incr = 0.4 / sqrt(layers)
    shell_list = np.linspace(1.4, 1.4 + layers * shell_incr, layers)
    
    # Get molecule connectivity data
    connectivity_index, connectivity_table = get_connectivity(atoms, np.array(x), np.array(y), np.array(z))
    angles_table, angles_index = find_angle_atoms(connectivity_index)
    
    
    # Write mol2 file for resp program
    mol2 = open(basename + "_mep.mol2", "w")
    mol2.write("# mol2 file generated for RESP calculations\n")
    mol2.write("\n")
    mol2.write("\n")
    mol2.write("@<TRIPOS>MOLECULE\n")
    mol2.write("Molecule name\n")
    mol2.write("{0:0d}{1:5d}\n".format(len(x), len(connectivity_index)))
    mol2.write("SMALL\n")
    mol2.write("NO_CHARGES\n")
    mol2.write("\n")
    mol2.write("\n")
    mol2.write("@<TRIPOS>ATOM\n")
    for i in range(0, len(x)):
        index = str(i +1 )
        mol2.write("{} {}  {}   {}    {}  {}\n".format(i + 1, atoms[i] + index, x[i], y[i], z[i], atoms[i]))
    mol2.write("@<TRIPOS>BOND\n")
    for i in range(0, len(connectivity_index)):
        mol2.write("{} {} {} {}  \n".format(i + 1, connectivity_index[i,0] + 1, connectivity_index[i,1] + 1, 1))
    mol2.close()
    
    
    
    #Generate MOL2 file to get AMBER atom types
    subprocess.run([antechamber_path + "antechamber", "-i", "orca_mep.mol2", "-fi", "mol2", "-o", "atom_names.ac", "-fo", "ac", "-nc", str(charge), "-pf", "y", "-dr", "no","-at", "amber"])

    # Read in AMBER atom names for use in output table
    atom_names = []
    f = open("atom_names.ac", "r")
    f.readline()
    f.readline()
    for i in range(len(atoms)):
        data = f.readline().split()
        if data[-1] != "DU":
            atom_names.append(data[-1])
        else:
            atom_names.append(atoms[i].upper())
    f.close()
    
    # Update connectivity_table with AMBER atom names
    for i in range(1, len(connectivity_table)):
        connectivity_table[i,0] = atom_names[int(connectivity_table[i,1])]
        connectivity_table[i,2] = atom_names[int(connectivity_table[i,3])]
    for i in range(1, len(angles_table)):
        angles_table[i,0] = atom_names[int(angles_table[i,1])]
        angles_table[i,2] = atom_names[int(angles_table[i,3])]
        angles_table[i,4] = atom_names[int(angles_table[i,5])]


        
    



    # RESP calcuation 
    if my_namespace.resp == "on":
        # Generate input file for orca_vpot
        gen_esp_points(x, y, z, npoints)
        # Run orca_vpot. NB! Check path to orca
        subprocess.run([orca_path + "orca_vpot", basename + ".gbw", basename + ".scfp", basename + "_mep.inp", basename + "_mep.out"])
        # Read result from orca_vpot
        vx, vy, vz, v = read_vpot(basename + "_mep.out")
        # Write *.esp file that can be read by resp program.
        esp = open(basename + "_mep.esp", "w")
        esp.write("   {}{}    {}\n".format(len(atoms),len(vx), 0))   #  "{:.2E}"
        for i in range(0, len(x)):
            esp.write("{0:32.7E}{1:16.7E}{2:16.7E}\n".format(x[i] * ang_to_au, y[i] * ang_to_au, z[i] * ang_to_au))
        for i in range(0, len(vx)):
            esp.write("{0:16.7E}{1:16.7E}{2:16.7E}{3:16.7E}\n".format(v[i], vx[i], vy[i], vz[i]))
        esp.close()
        # Run resp program
        
        if my_namespace.respfile == "None":
            
            subprocess.run([antechamber_path + "antechamber", "-i", "orca_mep.mol2", "-fi", "mol2", "-o", "mol.ac", "-fo", "ac", "-nc", str(charge), "-pf", "y", "-dr", "no"])
            subprocess.run([antechamber_path + "respgen", "-i", "mol.ac", "-o", "mol.respin1", "-f", "resp1"])
            subprocess.run([antechamber_path + "respgen", "-i", "mol.ac", "-o", "mol.respin2", "-f", "resp2"])
            subprocess.run([antechamber_path + "resp", "-O", "-i", "mol.respin1", "-o", "mol.respout1", "-e", "orca_mep.esp", "-t", "qout_stage1"])
            subprocess.run([antechamber_path + "resp", "-O", "-i", "mol.respin2", "-o", "mol.respout2", "-e", "orca_mep.esp", "-q", "qout_stage1", "-t", "qout_stage2"])
            subprocess.run([antechamber_path + "antechamber", "-i", "mol.ac", "-fi", "ac", "-o", "mol_resp.ac", "-fo", "ac", "-c", "rc", "-cf", "qout_stage2", "-dr", "no", "-pf", "y"])
            subprocess.run([antechamber_path + "antechamber", "-i", "mol_resp.ac", "-fi", "ac", "-o", "pyparam_resp_charges.mol2", "-fo", "mol2", "-dr", "no","-pf", "y","-at", "amber"])
            
            subprocess.run([antechamber_path + "antechamber", "-i", "mol_resp.ac", "-fi", "ac", "-o", str(my_namespace.name) +".mol2", "-fo", "mol2", "-dr", "no","-pf", "y","-at", "gaff2"])
            subprocess.run([antechamber_path + "parmchk2", "-i", str(my_namespace.name) +".mol2", "-f", "mol2", "-o", str(my_namespace.name) +".frcmod"])

            # Define the content with the 'MOL' string dynamically replaced
            content = f"""source leaprc.gaff2
            {str(my_namespace.name)} = loadmol2 {str(my_namespace.name)}.mol2
            loadamberparams {str(my_namespace.name)}.frcmod
            saveoff {str(my_namespace.name)} {str(my_namespace.name)}.lib
            quit"""
            
            # Write the content to a file named 'input_tleap'
            with open('input_tleap', 'w') as file:
                file.write(content)
                        
            subprocess.run([antechamber_path + "tleap", "-f", "input_tleap"])

        
        if my_namespace.respfile != "None":
            subprocess.run([antechamber_path + "antechamber", "-i", "orca_mep.mol2", "-fi", "mol2", "-o", "mol.ac", "-fo", "ac", "-nc", str(charge), "-pf", "y", "-dr", "no"])
            subprocess.run([antechamber_path + "respgen", "-i", "mol.ac", "-a", str(my_namespace.respfile), "-o", "mol.respin1", "-f", "resp1"])
            subprocess.run([antechamber_path + "respgen", "-i", "mol.ac", "-o", "mol.respin2", "-f", "resp2"])
            subprocess.run([antechamber_path + "resp", "-O", "-i", "mol.respin1", "-o", "mol.respout1", "-e", "orca_mep.esp", "-t", "qout_stage1"])
            subprocess.run([antechamber_path + "resp", "-O", "-i", "mol.respin2", "-o", "mol.respout2", "-e", "orca_mep.esp", "-q", "qout_stage1", "-t", "qout_stage2"])
            subprocess.run([antechamber_path + "antechamber", "-i", "mol.ac", "-fi", "ac", "-o", "mol_resp.ac", "-fo", "ac", "-c", "rc", "-cf", "qout_stage2", "-dr", "no", "-pf", "y"])
            subprocess.run([antechamber_path + "antechamber", "-i", "mol_resp.ac", "-fi", "ac", "-o", str(my_namespace.o), "-fo", "mol2", "-dr", "no","-pf", "y","-at", "amber"])
 
    
    if my_namespace.force == "on":
        ##### Tabulate  information and write to file ###########
        # Read Hessian matrix from the file orca.hess
        Hessian = read_hessian(natoms)     

        fc_b = [['Force constant upper', 'Force constant lower']]
        for i in range(0, len(connectivity_index)):
            fc_b.append([calc_bond_fc_up(xyz, Hessian, connectivity_index[i,0], connectivity_index[i,1]), calc_bond_fc_low(xyz, Hessian, connectivity_index[i,0], connectivity_index[i,1])])
            connectivity_table[i,0]
        bond_fc_table = np.c_[connectivity_table, fc_b]
    
        # Tabulate angle information
        fc_a = ['Force constant']
        for i in range(0, len(angles_index)):
            fc_a.append(calc_angle_fc(xyz, Hessian, angles_index[i,0], angles_index[i,1], angles_index[i,2]))
        angle_fc_table = np.c_[angles_table, fc_a]

        df = pd.DataFrame(bond_fc_table)
        df2 = pd.DataFrame(angle_fc_table)

        df.to_csv("PyForce.out",sep='\t', index=False, header=False)

        with open("PyForce.out", 'a') as f:
            df2.to_csv(f, sep='\t', index=False, header=False)
    
        #########################################################


        ##### Write template FRCMOD file for selected atom name ########################
        frcmod = open("frcmod-" + my_namespace.at + ".dat", "w")
        frcmod.write("Force constants generated by ORCA calculation. Based on Seminaro 1996.\n")
        frcmod.write("MASS\n")
        frcmod.write("Fill inn mases if nessecary, e.g. CU 63.55\n")
        frcmod.write("\n")
        frcmod.write("BOND\n")
        for i in range(1, len(bond_fc_table)):
            if (bond_fc_table[i,0] == my_namespace.at or bond_fc_table[i,2] == my_namespace.at):
                frcmod.write("{0:2s}-{1:2s}{2:15.1f}{3:15.4f}       # Alt. Hessian int. mat. {4:15.1f}\n".format(bond_fc_table[i,0], bond_fc_table[i,2], float(bond_fc_table[i,5]), float(bond_fc_table[i,4]), float(bond_fc_table[i,6])))
        frcmod.write("\n")
        frcmod.write("ANGLE\n")
        for i in range(1, len(angle_fc_table)):
            if (angle_fc_table[i,0] == my_namespace.at or angle_fc_table[i,2] == my_namespace.at):
                frcmod.write("{0:2s}-{1:2s}-{2:2s}{3:15.1f}{4:15.4f}\n".format(angle_fc_table[i,0], angle_fc_table[i,2], angle_fc_table[i,4], float(angle_fc_table[i,7]), float(angle_fc_table[i,6])))
        frcmod.close()

        ##### Write template FRCMOD file for ALL atoms ########################
        frcmod = open("frcmod-ALL.dat", "w")
        frcmod.write("Force constants generated by ORCA calculation. Based on Seminaro 1996.\n")
        frcmod.write("MASS\n")
        frcmod.write("Fill inn mases if nessecary, e.g. CU 63.55\n")
        frcmod.write("\n")
        frcmod.write("BOND\n")
        for i in range(1, len(bond_fc_table)):
            frcmod.write("{0:2s}-{1:2s}{2:15.1f}{3:15.4f}        # Alt. Hessian int. mat.  {4:15.1f}\n".format(bond_fc_table[i,0], bond_fc_table[i,2], float(bond_fc_table[i,5]), float(bond_fc_table[i,4]), float(bond_fc_table[i,6])))
        frcmod.write("\n")
        frcmod.write("ANGLE\n")
        for i in range(1, len(angle_fc_table)):
            frcmod.write("{0:2s}-{1:2s}-{2:2s}{3:15.1f}{4:15.4f}\n".format(angle_fc_table[i,0], angle_fc_table[i,2], angle_fc_table[i,4], float(angle_fc_table[i,7]), float(angle_fc_table[i,6])))
        frcmod.close()
        
        
        ##### CLEAN UP ################################
        
        import os
import shutil

# Define the folder names
gaff_ff_folder = "gaff2_ff"
seminaro_ff_folder = "seminaro_ff"

# Create the folders if they don't already exist
os.makedirs(gaff_ff_folder, exist_ok=True)
os.makedirs(seminaro_ff_folder, exist_ok=True)

# Define the files to move to 'gaff_ff' folder
gaff_files = [
    str(my_namespace.name)+".frcmod",
    str(my_namespace.name)+".lib",
    str(my_namespace.name)+".mol2"
]

# Move files to 'gaff_ff' folder using a loop
for file in gaff_files:
    shutil.move(file, os.path.join(gaff_ff_folder, file))

# Define the files to move to 'seminaro_ff' folder
seminaro_files =  [
    "PyForce.out",
    "frcmod-ALL.dat",
    "frcmod-None.dat",
    "pyparam_resp_charges.mol2"
]

# Move files to 'seminaro_ff' folder using a loop
for file in seminaro_files:
    shutil.move(file, os.path.join(seminaro_ff_folder, file))

# Define the files to delete
files_to_delete = files = [
    "ESP_points_for_visualization.xyz",
    "atom_names.ac",
    "esout",
    "input_tleap",
    "leap.log",
    "mol.ac",
    "mol.respin1",
    "mol.respin2",
    "mol.respout1",
    "mol.respout2",
    "mol_resp.ac",
    "orca_mep.esp",
    "orca_mep.inp",
    "orca_mep.mol2",
    "orca_mep.out",
    "punch",
    "qout_stage1",
    "qout_stage2"
]

# Delete files using a loop
for file in files_to_delete:
    os.remove(file)

    