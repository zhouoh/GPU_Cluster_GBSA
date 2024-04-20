import os
import sys
#command line arguments
import argparse
import numpy as np
import base64
import csv

IN_base64 = "/Td6WFoAAATm1rRGAgAhARYAAAB0L+Wj4E//Ay9dADKcQiFazHbptEya0uKdmUo0mSwYH4POAs73jXL0P1XFo0TSna3Iorhyo3hw/Py7Ar1hoxpWKsai7KXZT5sDsH6vDjzkUBOZZtCeqsjQUc53ohkR1XT4cAlXgoMKlocMIjvHEaMXJxPKeS8uCvksJTl5mR6Nyxdo5JcXy6Xo5l5Bij8QHCCZ9yQ38lWpzqAM4BDng8whUgh8viaJBZNa3nLOQfPJZ4hhWt2W7Fs+qxaAY6THeyB5sAJFroKS7jFHxvXVlTEXZiySceitPoxTfTAK68DT78tGUJxzM5cq1zU8uwFHCsvMkPtiVQ7CkSI499Mkj/Wp1OdUurcDoL6mryryRaMzOwHJywp6ctBcF5RZcxeVk9dy2f1JWx5okTGwnjOqM/5qm9RSBQ2PC5ENHD1lSUxJEA/uQdBYKRDzGYjtwqTKSA5Ew3MHE4gb15uV7JyvwIGo9EsvQKgU11ijLEszj7V+tgXrv0QvEBVaGUE4iy99/BvmKfunjaSWPZ4qZDSwM9FTy/hotYq0Y0Vqr8IXCE1oVbUhbnTtlWfxoY7GKnpdLlWY3kxC3NvYVTih1LDADXU0UUUl+ql6eRLtRz/4fd9TZX4P6eavW33yJazl/pcZi4rAOQnZ9xHHBZzWm3giMrQBLS5C8t76RMrj7c32kHigjuKS6J4sUkmwbtHXMHJAM1hCIh2Tpo+prEWuwaDiqGmPHG7LQ4EGXHq5v2RrI97TMWOdxPiSmuiCEbSayMpTvUTu+MAXBlfUCxsXPf6+z59JvtfHORXcMxdI+y7hJ0OaHl4XTBvm9tDWHF7s5YK3vbs0Q0ycum4YfQPIqVGO8k2MAlg57qdrGwt5F2cfgAjxM2Iv4Sf/diFSYyXovganuBux42iv+0TEuU69tVk4D+WOCvziKFCRFScCUuYNbKcrl4r5jDH9Rhuiud36OBraNk3lVbDDWBZWVb73fPYa8WIfwtxNyIpxTRyQGwQsSEigD40OHzIb90bUGw8CNfCFbpnCi5kWkIQ8wbRXpY0t7368NObyW7n0sgB36R/oK0J3aCb8z/OgBbzoI1CAwrxPUapUDu5zJRyBsqDAAAALHBjwkAZF+QABywaAoAEAEBUAGrHEZ/sCAAAAAARZWg=="
EXCLUDE_GPU_ID = 8
IF_RESTART = False

def check_gpu_status():
    idle_cutoff = 20
    gpu_status = []
    #check if the GPU is idle using nvidia-smi
    os.system('nvidia-smi --query-gpu=index,name,utilization.gpu --format=csv > nvidia-smi.csv')
    with open('nvidia-smi.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            utilization = int(row[2].split()[0])
            if utilization < idle_cutoff:
                gpu_status.append(True)
            else:
                gpu_status.append(False)
    return gpu_status

def write_submission_script_body(f):
    f.write(f'CUDA_VISIBLE_DEVICES=$device $command -O -i min1.in  -o min1.out   -p *prmtop -c *.inpcrd -r min1.rst   -x min1.nc   -ref *.inpcrd\n')
    f.write(f'CUDA_VISIBLE_DEVICES=$device $command -O -i min2.in  -o min2.out   -p *prmtop -c min1.rst -r min2.rst   -x min2.nc   -ref min1.rst\n')
    f.write(f'CUDA_VISIBLE_DEVICES=$device $command -O -i heat1.in  -o heat1.out   -p *prmtop -c min2.rst  -r heat1.rst   -x heat1.nc   -ref min2.rst\n')
    f.write(f'CUDA_VISIBLE_DEVICES=$device $command -O -i heat2.in  -o heat2.out   -p *prmtop -c heat1.rst  -r heat2.rst   -x heat2.nc   -ref heat1.rst\n')
    f.write(f'CUDA_VISIBLE_DEVICES=$device $command -O -i heat3.in  -o heat3.out   -p *prmtop -c heat2.rst  -r heat3.rst   -x heat3.nc   -ref heat2.rst\n')
    f.write(f'CUDA_VISIBLE_DEVICES=$device $command -O -i heat4.in  -o heat4.out   -p *prmtop -c heat3.rst  -r heat4.rst   -x heat4.nc   -ref heat3.rst\n')
    f.write(f'CUDA_VISIBLE_DEVICES=$device $command -O -i heat5.in  -o heat5.out   -p *prmtop -c heat4.rst  -r heat5.rst   -x heat5.nc   -ref heat4.rst\n')
    f.write(f'CUDA_VISIBLE_DEVICES=$device $command -O -i heat6.in  -o heat6.out   -p *prmtop -c heat5.rst  -r heat6.rst   -x heat6.nc   -ref heat5.rst\n')
    f.write(f'CUDA_VISIBLE_DEVICES=$device $command -O -i eq1.in  -o eq1.out   -p *prmtop -c heat6.rst  -r eq1.rst   -x eq1.nc   -ref heat6.rst\n')
    f.write(f'CUDA_VISIBLE_DEVICES=$device $command -O -i eq2.in  -o eq2.out   -p *prmtop -c eq1.rst  -r eq2.rst   -x eq2.nc   -ref eq1.rst\n')
    f.write(f'CUDA_VISIBLE_DEVICES=$device $command -O -i prod.in -o prod.out -p *prmtop -c eq2.rst  -r prod.rst -x prod.nc\n')
    #f.write('#### submit_job.sh END ####\n')

def write_submission_script_body_restart(f):
    f.write(f'CUDA_VISIBLE_DEVICES=$device $command -O -i prod.in -o prod.out -p *prmtop -c eq2.rst  -r prod.rst -x prod.nc\n')


def write_submission_script_header(f,gpu_id):
    f.write('#### submit_job.sh START ####\n')
    f.write('\n')
    f.write('source /home/xucongs/software/amber22/amber.sh\n')
    f.write('command=pmemd.cuda\n')
    f.write(f"device={gpu_id}")
    f.write('\n')

def make_work_dir(name,num_gpu,num_replica,index,index_l,length,gpu_status):
    #print current working directory
    cwd = os.getcwd()
    print(f'Current working directory: {cwd}')
    #calculate number of replicas per GPU
    num_replica_per_gpu = int(num_replica / num_gpu)
    print(f'Replicas per GPU: {num_replica_per_gpu}')
    #warn if there remain replicas
    if num_replica % num_gpu != 0:
        print(f'Warning: cannot evenly distribute {num_replica} replicas over {num_gpu} GPUs')
    #create a 2D list 
    replica = []
    for i in range(num_gpu):
        replica.append([])
    #check if the GPU is idle
    gpu_list = []
    for i in range(num_gpu):
        for j in gpu_status:
            if j and j != EXCLUDE_GPU_ID and i not in gpu_list:
                gpu_list.append(i)
                break
    #check if the number of idle GPUs is enough
    if len(gpu_list) < num_gpu:
        print(f'Error: not enough idle GPUs')
        return
    
    #create work directory
    for i in range(num_replica):
        #determine which GPU to use
        gpu = int(i / num_replica_per_gpu)
        if gpu >= num_gpu:
            gpu = num_gpu - 1
        #create directory
        dir_name = f'{name}_replica_{i}_gpu_{gpu_list[gpu]}'
        print(f'Creating directory: {dir_name}')
        replica[gpu].append(dir_name)
        try:
            os.mkdir(dir_name)
        except OSError:
            #print(f'Creation of the directory {dir_name} failed')
            pass
        #copy files
        os.system(f'cp *.prmtop {dir_name}')
        os.system(f'cp *.inpcrd {dir_name}')
        os.chdir(dir_name)
        #write input files
        with open("in.tar.xz", "wb") as f:
            f.write(base64.b64decode(IN_base64))
        os.system('tar -xf in.tar.xz')
        modifiy_input(index,index_l,length)
        os.system('rm in.tar.xz')
        os.chdir("..")
    return replica, gpu_list

def write_submission_script(replica,gpu_list):
    #create submission script
    submission_script_list = []
    for i in range(len(replica)):
        gpu_id = gpu_list[i]
        submission_script_list.append(f'submit_job_gpu_{gpu_id}.sh')    
        with open(f'submit_job_gpu_{gpu_id}.sh','w') as f:
            write_submission_script_header(f,gpu_id)
            for dir_name in replica[gpu_id]:
                f.write(f'cd {dir_name}\n')
                if IF_RESTART == False:
                    write_submission_script_body(f)
                else:
                    write_submission_script_body_restart(f)
                f.write(f'cd ..\n')
            f.write('#### submit_job.sh END ####\n')
        os.system(f'chmod +x submit_job_gpu_{gpu_id}.sh')
    return submission_script_list

def modifiy_input(index,index_l,length):
    #list all the .in files in the current directory
    in_files = [f for f in os.listdir('.') if f.endswith('.in')]
    #modify the input files
    for i in in_files:
        #print(f'Modifying {i}')
        content = ""
        with open(i,'r') as f:
            content = f.read()
        #replace the index
        content = content.replace("[index]",index)
        content = content.replace("[index_l]",index_l)
        with open(i,'w') as f:
            f.write(content)
    with open('prod.in','r') as f:
        content = f.read()
    #convert ns to steps
    length = int(float(length) * 50000)
    d = int(length / 100)
    length = str(length)
    d = str(d)
    content = content.replace("[length]",length)
    content = content.replace("[d]",d)
    with open('prod.in','w') as f:
        f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create submission script for AMBER simulation')
    parser.add_argument('--num_gpu', type=int, help='number of GPUs', default=4)
    parser.add_argument('--num_replica', type=int, help='number of replicas', default=20)
    parser.add_argument('--index', type=str, help='index of the last protein residue', default='281')
    parser.add_argument('--index_l', type=str, help='index of the ligand', default='282')
    parser.add_argument('--submit', type=bool, help='If submit', default=False)
    parser.add_argument('--restart', action='store_true', help='If restart')
    parser.add_argument('--length', type=float, help='Length of the simulation in ns', default=5.0)
    args = parser.parse_args()
    if args.restart:
        IF_RESTART = True
    replica, gpu_list = make_work_dir(args.name,args.num_gpu,args.num_replica,args.index,args.index_l,args.length,check_gpu_status())
    submission_list = write_submission_script(replica,gpu_list)
    if args.submit:
        for i in submission_list:
            os.system(f'nohup bash ./{i} &')
    else:
        print('Not submitting jobs')


   

