n=36462
for (( i=$n ; i< 1000000 ; i++ )); 
do
    rm /home/yixuany/workspace/CORL/finetune_data_sac/data_$i.pkl
done