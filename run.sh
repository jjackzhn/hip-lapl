export OMP_NUM_THREADS=4
L="256,256 512,256 512,512 1024,512 1024,1024 2048,1024 2048,2048 4096,2048 4096,4096 8192,4096 8192,8192"
for x in $L ; do
    Lx=$(echo $x|cut -d, -f2)
    Ly=$(echo $x|cut -d, -f1)
    python mkinit.py $Lx $Ly ini
    ./main $Lx $Ly 1000 ini o
done
