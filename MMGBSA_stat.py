import os
import sys
#command line arguments
import argparse
import numpy as np
import re

IF_OVERWRITE = False

def write_GBSA_input(f):
    f.write('&general\n')
    f.write('startframe=1, endframe=100, interval=1,\n') 
    f.write('verbose=2, keep_files=0,\n')
    f.write('/\n')
    f.write('&gb\n') 
    f.write('igb=1, saltcon=0.150,\n')
    f.write('/\n')
    f.write('&pb\n')
    f.write('istrng=0.15, fillratio=4.0, indi=4.0, \n')
    f.write('/\n')

def write_recomannd_GBSA_input(f):
    f.write('&general\n')
    f.write('startframe=1, endframe=100, interval=1,\n') 
    f.write('verbose=2, keep_files=0,use_sander=1\n')
    f.write('/\n')
    f.write('&gb\n')
    f.write('igb=1, saltcon=0.150,\n')
    f.write('/\n')


def write_recomannd_PBSA_input(f):
    f.write('&general\n')
    f.write('startframe=1, endframe=100, interval=1,\n') 
    f.write('verbose=2, keep_files=0,\n')
    f.write('/\n')
    f.write('&pb\n')
    f.write('istrng=0.15, fillratio=4.0, indi=1.0, \n')
    f.write('/\n')

def write_QMMM_GBSA_input(f,qm_residue,charge):
    f.write('&general\n')
    f.write('startframe=1, endframe=100, interval=1,\n') 
    f.write('verbose=2, keep_files=0,use_sander=1\n')
    f.write('/\n')
    f.write('&gb\n')
    f.write(f'igb=1, saltcon=0.150, ifqnt=1, qmcharge_lig=0, qm_residues="{qm_residue}", qm_theory="PM6",qmcharge_com={charge}, qmcharge_rec={charge}\n')
    f.write('/\n')
    
   

def get_dirs():
    #list all the directories in the current working directory
    dirs = os.listdir()
    dirs = [d for d in dirs if os.path.isdir(d)]
    # exclude the directories that not contain the "replica" string
    dirs = [d for d in dirs if "replica" in d]
    print(f'Detected directories: {dirs}')
    num_replica = len(dirs)
    replica_list = []
    #initialize the list of replica
    for i in range(num_replica):
        replica_list.append("")
    for d in dirs:
        #determine the index of the replica
        index = int(d.split('_')[2])
        replica_list[index] = d
    return replica_list

def prepare_prm(index_l):
    #list the .prmtop files in the current directory
    prmtop_files = [f for f in os.listdir('.') if f.endswith('.prmtop')]
    if len(prmtop_files) == 0:
        print("Error: no .prmtop file found")
        sys.exit(1)
    elif len(prmtop_files) > 1:
        print("Error: multiple .prmtop files found")
        sys.exit(1)
    prmtop_raw = prmtop_files[0]
    strip_mask = ":WAT,Na+,Cl-"
    ligand_mask = f":{index_l}"
    print(f'Using prmtop file: {prmtop_raw}')
    print(f'Runing ante-MMPBSA.py -p {prmtop_raw} -c com.parm -r rec.parm -l lig.parm -s {strip_mask} -n {ligand_mask}')
    os.system(f'ante-MMPBSA.py -p {prmtop_raw} -c com.parm -r rec.parm -l lig.parm -s {strip_mask} -n {ligand_mask}')

def run_MMGBSA(replica_dir):
    os.chdir(replica_dir)
    prmtop_files = [f for f in os.listdir('.') if f.endswith('.prmtop')]
    prmtop_raw = prmtop_files[0]
    #check if we already have the output file
    if os.path.isfile('gbsa.out') and not IF_OVERWRITE:
        with open('gbsa.out', 'r') as f:
            content = f.read()
            if "DELTA TOTAL" in content:
                print(f'GBSA output file already exists in {replica_dir}')
                os.chdir('..')
                return
    with open('gbsa.in', 'w') as f:
        write_GBSA_input(f)
    os.system(f'MMPBSA.py -O -i gbsa.in -o gbsa.out -sp {prmtop_raw} -cp ../com.parm -rp ../rec.parm -lp ../lig.parm -y prod.nc > gbsa.log')
    os.chdir('..')

def run_recommand_MMGBSA(replica_dir):
    os.chdir(replica_dir)
    prmtop_files = [f for f in os.listdir('.') if f.endswith('.prmtop')]
    prmtop_raw = prmtop_files[0]
    #check if we already have the output file
    if os.path.isfile('gbsa_2.out') and not IF_OVERWRITE:
        with open('gbsa_2.out', 'r') as f:
            content = f.read()
            if "DELTA TOTAL" in content:
                print(f'GBSA output file already exists in {replica_dir}')
                os.chdir('..')
                return
    with open('gbsa.in', 'w') as f:
        write_recomannd_GBSA_input(f)
    os.system(f'MMPBSA.py -O -i gbsa.in -o gbsa_2.out -sp {prmtop_raw} -cp ../com.parm -rp ../rec.parm -lp ../lig.parm -y prod.nc -make-mdins > gbsa_2.log')
    with open('_MMPBSA_gb.mdin', 'r') as f:
        content = f.read()
        content = content.replace('extdiel=80.0,', 'extdiel=80.0, intdiel=4.0,')
    with open('_MMPBSA_gb.mdin', 'w') as f:
        f.write(content)
    os.system(f'MMPBSA.py -O -i gbsa.in -o gbsa_2.out -sp {prmtop_raw} -cp ../com.parm -rp ../rec.parm -lp ../lig.parm -y prod.nc -use-mdins > gbsa_2.log')
    with open("gbsa.in","w") as f:
        write_recomannd_PBSA_input(f)
    os.system(f'MMPBSA.py -O -i gbsa.in -o gbsa_2p.out -sp {prmtop_raw} -cp ../com.parm -rp ../rec.parm -lp ../lig.parm -y prod.nc > gbsa_2p.log')
    #combine the results
    with open('gbsa_2p.out', 'r') as f:
        content = f.read()
        with open('gbsa_2.out', 'a') as f2:
            f2.write("\n")
            f2.write(content)
    os.chdir('..')

def get_resid(index_l,cutoff):
    import MDAnalysis as mda
    prmtop_files = [f for f in os.listdir('..') if f.endswith('.prmtop')]
    prmtop_raw = prmtop_files[0]
    inpcrd_files = [f for f in os.listdir('..') if f.endswith('.inpcrd')]
    inpcrd_raw = inpcrd_files[0]
    print(f'Using prmtop file: {prmtop_raw}')
    print(f'Using inpcrd file: {inpcrd_raw}')
    u = mda.Universe(prmtop_raw,inpcrd_raw)
    #select all the protein residues around cutoff of ligand
    res = u.select_atoms(f'(around {cutoff} resid {index_l}) and protein')
    res = res.residues
    res = [r.resid for r in res]
    res_str = [str(r) for r in res]
    res_str = ','.join(res_str)
    print(f'QM residues: {res_str}')
    total_charge = 0
    for r in res:
        #print(r)
        sel = u.select_atoms(f'resid {r}')
        total_charge += np.sum(sel.charges)
    print(f'Total charge: {total_charge}')
    return res_str, int(round(total_charge))



def run_QMMMGBSA(replica_dir,index_l,cutoff,residue):
    os.chdir(replica_dir)
    prmtop_files = [f for f in os.listdir('.') if f.endswith('.prmtop')]
    prmtop_raw = prmtop_files[0]
    #check if we already have the output file
    if os.path.isfile('gbsa_3.out') and not IF_OVERWRITE:
        with open('gbsa_3.out', 'r') as f:
            content = f.read()
            if "DELTA TOTAL" in content:
                print(f'GBSA output file already exists in {replica_dir}')
                os.chdir('..')
                return
    if residue != "12138":
        qm_residue = residue
    else:
        qm_residue, charge = get_resid(index_l,cutoff)
        qm_residue += f',{index_l}'
    with open('gbsa.in', 'w') as f:
        write_QMMM_GBSA_input(f,qm_residue,charge)
    os.system(f'MMPBSA.py -O -i gbsa.in -o gbsa_3.out -sp {prmtop_raw} -cp ../com.parm -rp ../rec.parm -lp ../lig.parm -y prod.nc > gbsa_3.log')
    os.chdir('..')


def extract_GBSA_results(replica_list,method):
    GBSA_dG = []
    PBSA_dG = []
    if method == 1:
        outfile = 'gbsa.out'
    elif method == 2:
        outfile = 'gbsa_2.out'
    elif method == 3:
        outfile = 'gbsa_3.out'
    for replica in replica_list:
        os.chdir(replica)
        with open(outfile, 'r') as f:
            content = f.read()
        dG = re.findall(r'DELTA TOTAL\s+(-?\d+\.\d+)', content)
        GBSA_dG.append(float(dG[0]))
        try:
            PBSA_dG.append(float(dG[1]))
        except:
            PBSA_dG.append(float(dG[0]))
        os.chdir('..')
    return GBSA_dG, PBSA_dG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run and Extract MMGBSA results')
    parser.add_argument('--index_l', type=str, help='index of the ligand',default="282")
    parser.add_argument('-N', '--num_cpu', type=int, help='number of CPU', default=12138)
    parser.add_argument('-o', '--output', type=str, help='output file', default='GBSA_result.dat')
    parser.add_argument('-c','--cutoff', type=float, help='cutoff for QM region', default=2.5)
    parser.add_argument('-r','--residue', type=str, help='residue for QM region', default="12138")
    parser.add_argument('-m','--method', type=int, help='method for GB(PB) calculation\n1. normal GB(HCT) and PB (epsilon=4.0)\n2. GB(epsilon=4.0) and PB (epsilon=1.0)\n 3. QM/MM-GBSA using PM6', default=1)
    parser.add_argument('-O','--overwrite', action='store_true', help='overwrite the existing output file', default=False)


    args = parser.parse_args()
    if args.num_cpu == 12138:
        args.num_cpu = os.cpu_count()/2
        print(f'Using {args.num_cpu} cores')
    #Throw a warning if too many cores are used
    if args.num_cpu > os.cpu_count()/2:
        print(f"Warning: using more than {os.cpu_count()/2} cores may cause the system to crash")
    args.num_cpu = int(args.num_cpu) 
    IF_OVERWRITE = args.overwrite

    replica_list = get_dirs()
    prepare_prm(args.index_l)
    #Open a pool of workers
    from multiprocessing import Pool
    if args.method == 1:
        with Pool(args.num_cpu) as p:
            p.map(run_MMGBSA, replica_list)
        GBSA_dG, PBSA_dG = extract_GBSA_results(replica_list,args.method)
        import prettytable as pt
        table = pt.PrettyTable(["Replica", "GBSA dG", "PBSA dG"])
        for i in range(len(replica_list)):
            table.add_row([f"{i+1}", f"{GBSA_dG[i]:.2f}", f"{PBSA_dG[i]:.2f}"])
        avg_g = np.mean(GBSA_dG)
        avg_p = np.mean(PBSA_dG)
        table.add_row(["Average", f"{avg_g:.2f}", f"{avg_p:.2f}"])
        print(table)
        #also save a dat file
        with open(args.output, 'w') as f:
            f.write(table.get_string())
    elif args.method == 2:
        with Pool(args.num_cpu) as p:
            p.map(run_recommand_MMGBSA, replica_list)
        GBSA_dG, PBSA_dG = extract_GBSA_results(replica_list,args.method)
        import prettytable as pt
        table = pt.PrettyTable(["Replica", "GBSA dG", "PBSA dG"])
        for i in range(len(replica_list)):
            table.add_row([f"{i+1}", f"{GBSA_dG[i]:.2f}", f"{PBSA_dG[i]:.2f}"])
        avg_g = np.mean(GBSA_dG)
        avg_p = np.mean(PBSA_dG)
        table.add_row(["Average", f"{avg_g:.2f}", f"{avg_p:.2f}"])
        print(table)
        #also save a dat file
        with open(args.output, 'w') as f:
            f.write(table.get_string())
    elif args.method == 3:
        with Pool(args.num_cpu) as p:
            p.starmap(run_QMMMGBSA, [(replica_list[i],args.index_l,args.cutoff,args.residue) for i in range(len(replica_list))])
        GBSA_dG, _ = extract_GBSA_results(replica_list,args.method)
        #print(GBSA_dG)
        import prettytable as pt
        table = pt.PrettyTable(["Replica", "GBSA dG"])
        for i in range(len(replica_list)):
            table.add_row([f"{i+1}", f"{GBSA_dG[i]:.2f}"])
        avg_g = np.mean(GBSA_dG)
        table.add_row(["Average", f"{avg_g:.2f}"])
        print(table)
        #also save a dat file
        with open(args.output, 'w') as f:
            f.write(table.get_string())

