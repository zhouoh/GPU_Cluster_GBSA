import os
import sys
#command line arguments
import argparse
import numpy as np
import base64

IN_base64 = "H4sIABO2K2YAA+2bQW+bMBiGc+6v8C7NhXa2A2SqFKlTV2mXVT1Ek6YqqmhCE2tgKJglq/bjZ0OSAiFAooYQ9Xt7iN28mA9qP59Njf1CLhnvHFRYytR19UmIbqQ/lYjepx1Z7+s6oUT5CDGI3kH4sGElikJhBQh1Xmde5M22+6q+P1FRHiL7JWIOewoswTyOGEd390NkCTRj0xkK7FAEFuMCKZ2h8zEXgYNq6gwh5soWERogrCGmWpNFoiEuFrJgaBt+LsYo9lNlek4KJe2PI5H4v1zi7cZ0+0t/T0NTy3WtR4erSAqPln5huz6L/Rgrj6rjVH2z/adU/H5ytU9W4MmOJpJfCyvyB/EJpd+X9yRuj9SLn80Dy0dJs5VS/ilK4rmocYDye5F4duu3z0PhsNifDGcNTdRlynuDN/9w8f3xgzgeasRuLuaLTC14qyX+IIlfxrPujI9zEf8F8rdM+tce1wp/S1P3ilw8MD6xF6PrG+3mq3b37yqpPzqj80/X3wddLT7yc/Xlps907JH7PrJfaMv4TxP+94D/TaiM/443z+Ef+A/8z/uPyH98WdB/gP+7aGZb4tArgCr+90me/xRT4H8jMrAfItUJGJ/G6P85LB/v78J/vOJ/njFt5n+W/sYm/FGW/3X53FaeGzmaFx2f53kG5xmaa2/+3ebz3A08X9SPvyn+o3MZ8/DX/e2gO7z9cY9lKywUtk8GeFmig+UtTPx/LCey5bfqGuOy+l5VCk+cbv/27lu3Krw985Ea+odeAezFf3j+04iOw/+TnP8b2QRACqb/wH/gfxX/jXQCWHaiglM3yP9eG/lvAP+bEPC/Nv9J7vkPKVgCAP+B/xX8X3ajVQLYtgRokP96G/lvAv+bEPC/Pv9zCwBa8fwf+A/8L+J/ZgFAty0AGuS/0Ub+94H/TQj4X5v/NDf/pzD/B/7vzn+amf/TFsz/zRbyv4eB/00I+F+f/7n5f9X+H+A/8L+I/5n5f+/I8385OA/+AkAV/6lu5vd/mibM/xsR40wwy0GyH0hQvyZbQEt7/d78J/X2/7jWYvx3vOSP9CWVN4Tk/et8QVb5ogQUe+WLE+f5is9pPKfpnL6z5Xw2NjJeKW8lX7uFMX7QvZZtlBycB38BoJL/xsb+f9OE//82IuB/VfzA/zX/adF6B/bbn7L8wJscm/89WSa6KfOAYRKzp/jf78P+z0ZEMOYhUr0gGqdf/9rKk4/7/Cfz/lfx459Mvvho7389ODafitmo7AWwdD56mIzW+WhVDlbl3fPDGdr6c+xBBgKBQCAQCAQCgUAg0JH1H45e1QcAUAAA"
IF_RESTART = False
LENGTH = 5.0

def write_submission_script_body(f):
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i min1.in  -o min1.out   -p *prmtop -c *.inpcrd -r min1.rst   -x min1.nc   -ref *.inpcrd\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i min2.in  -o min2.out   -p *prmtop -c min1.rst -r min2.rst   -x min2.nc   -ref min1.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i heat1.in  -o heat1.out   -p *prmtop -c min2.rst  -r heat1.rst   -x heat1.nc   -ref min2.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i heat2.in  -o heat2.out   -p *prmtop -c heat1.rst  -r heat2.rst   -x heat2.nc   -ref heat1.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i heat3.in  -o heat3.out   -p *prmtop -c heat2.rst  -r heat3.rst   -x heat3.nc   -ref heat2.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i heat4.in  -o heat4.out   -p *prmtop -c heat3.rst  -r heat4.rst   -x heat4.nc   -ref heat3.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i heat5.in  -o heat5.out   -p *prmtop -c heat4.rst  -r heat5.rst   -x heat5.nc   -ref heat4.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i heat6.in  -o heat6.out   -p *prmtop -c heat5.rst  -r heat6.rst   -x heat6.nc   -ref heat5.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i eq1.in  -o eq1.out   -p *prmtop -c heat6.rst  -r eq1.rst   -x eq1.nc   -ref heat6.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i eq2.in  -o eq2.out   -p *prmtop -c eq1.rst  -r eq2.rst   -x eq2.nc   -ref eq1.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i prod.in -o prod.out -p *prmtop -c eq2.rst  -r prod.rst -x prod.nc\n')
    #f.write('#### submit_job.sh END ####\n')

def write_submission_script_body_restart(f):
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i prod.in -o prod.out -p *prmtop -c eq2.rst  -r prod.rst -x prod.nc\n')

def write_submission_script_header(f,name,gpu):
    f.write('#### submit_job.sh START ####\n')
    f.write('#!/bin/bash\n')
    f.write('#$ -cwd\n')
    f.write(f'#$ -N {name}\n')
    f.write(f'#$ -q {gpu}\n')
    f.write('\n')
    f.write('export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64\n')
    f.write('export AMBERHOME=/usr/local/amber22/\n')
    f.write('\n')

def make_work_dir(name,num_gpu,num_replica,index,index_l):
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
    #create work directory
    for i in range(num_replica):
        #determine which GPU to use
        gpu = int(i / num_replica_per_gpu)
        if gpu >= num_gpu:
            gpu = num_gpu - 1
        #create directory
        dir_name = f'{name}_replica_{i}_gpu_{gpu}'
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
        if IF_RESTART:
            #chek if restart files exist
            if not os.path.exists('eq2.rst'):
                print(f'Error: restart files not found in {dir_name}')
                sys.exit(1)
        #write input files
        with open("in.tar.xz", "wb") as f:
            f.write(base64.b64decode(IN_base64))
        os.system('tar -xf in.tar.xz')
        modifiy_input(index,index_l)
        os.system('rm in.tar.xz')
        os.chdir("..")
    return replica

def write_submission_script(name,replica,gpu):
    #create submission script
    submission_script_list = []
    for gpu_id in range(len(replica)):
        submission_script_list.append(f'submit_job_gpu_{gpu_id}.sh')    
        with open(f'submit_job_gpu_{gpu_id}.sh','w') as f:
            write_submission_script_header(f,f'{name}_gpu_{gpu_id}',gpu)
            for dir_name in replica[gpu_id]:
                f.write(f'cd {dir_name}\n')
                if IF_RESTART:
                    write_submission_script_body_restart(f)
                else: 
                    write_submission_script_body(f)
                f.write(f'cd ..\n')
            f.write('#### submit_job.sh END ####\n')
        os.system(f'chmod +x submit_job_gpu_{gpu_id}.sh')
    return submission_script_list

def modifiy_input(index,index_l):
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
    length = LENGTH
    length = int(float(length) * 500000)
    d = int(length / 100)
    length = str(length)
    d = str(d)
    content = content.replace("[length]",length)
    content = content.replace("[d]",d)
    with open('prod.in','w') as f:
        f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create submission script for AMBER simulation')
    parser.add_argument('--name', type=str, help='job name', default='mmgbsa')
    parser.add_argument('--num_gpu', type=int, help='number of GPUs', default=4)
    parser.add_argument('--num_replica', type=int, help='number of replicas', default=20)
    parser.add_argument('--gpu', type=str, help='GPU queue', default='gtx2080ti_gpu.q')
    parser.add_argument('--index', type=str, help='index of the last protein residue', default='281')
    parser.add_argument('--index_l', type=str, help='index of the ligand', default='282')
    parser.add_argument('--submit', type=bool, help='If submit', default=False)
    parser.add_argument('--restart', action='store_true', help='If restart')
    parser.add_argument('--length', type=float, help='Length of the simulation in ns', default=5.0)
    args = parser.parse_args()
    IF_RESTART = args.restart
    LENGTH = args.length
    replica = make_work_dir(args.name,args.num_gpu,args.num_replica,args.index,args.index_l)
    submission_list = write_submission_script(args.name,replica,args.gpu)
    if args.submit:
        for i in submission_list:
            os.system(f'qsub {i}')
    else:
        print('Not submitting jobs')


   

