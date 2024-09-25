n=36212
for (( i=$n ; i< 1000000 ; i++ )); 
do
    rm /home/yixuany/workspace/CORL/finetune_data_sac_risk_measure/data_$i.pkl
done