for i in `seq 1 1 20`
do
    cd ~/frcnn_keras_original

    python script/csv2json.py --src_path /data3/sap/frcnn_keras/result/result-messytable_from_1041_model_$i/val/log.csv --dst_path /data3/sap/frcnn_keras/result/result-messytable_from_1041_model_$i/val/log.json 

    CUDA_VISIBLE_DEVICES=-1 python main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/sv_messytable_from_1041_$i --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-messytable_from_1041_model_$i/val/log.json 

done

: <<'END'
for i in `seq 100 1 2000`
do
    cd ~/frcnn_keras_original

    CUDA_VISIBLE_DEVICES=-1 python main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/sv_messytable_$i --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-messytable_model_$i/val/log.json 

done

for i in `seq 1012 1 2000`
do
    cd ~/frcnn_keras_original

    python script/csv2json.py --src_path /data3/sap/frcnn_keras/result/result-messytable_model_$i/val/log.csv --dst_path /data3/sap/frcnn_keras/result/result-messytable_model_$i/val/log.json 

    CUDA_VISIBLE_DEVICES=-1 python main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/sv_messytable_$i --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-messytable_model_$i/val/log.json 

done
END
