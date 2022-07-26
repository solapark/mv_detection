for i in `seq 0 1 40`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=1 python test.py  --save_name messytable_from_1041 --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar
done

: <<'END'
for i in `seq 1 1 20`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=1 python test.py  --save_name messytable_from_1041 --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar
done


for i in `seq 1285 2 2000`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=1 python test.py  --save_name messytable --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar
done


for i in `seq 1013 2 2000`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=1 python test.py  --save_name messytable --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar
done

for i in `seq 1001 2 2000`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=3 python test.py  --save_name messytable --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar
done

for i in `seq 528 3 1000`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=3 python test.py  --save_name messytable --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar
done

for i in `seq 529 3 1000`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=2 python test.py  --save_name messytable --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar
done

for i in `seq 530 3 1000`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=2 python test.py  --save_name messytable --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar
done



for i in `seq 489 2 1000`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=2 python test.py  --save_name messytable --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar
done


for i in `seq 488 2 1000`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=3 python test.py  --save_name messytable --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar
done

for i in `seq 300 1 1000`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=0 python test.py  --save_name messytable --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar
done

for i in `seq 351 2 1000`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=3 python test.py  --save_name messytable --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar
done

for i in `seq 350 2 1000`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=0 python test.py  --save_name messytable --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar
done

for i in `seq 180 1 1000`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=2 python test.py  --save_name messytable --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle --progbar

    cd ~/frcnn_keras_original

    CUDA_VISIBLE_DEVICES=-1 python main.py --mode val_json_json --reset --dataset MESSYTABLE --save_dir val_json_json/sv_messytable_$i --num_valid_cam 3 --dataset_path /data1/sap/MessyTable/labels/val.json --pred_dataset_path /data3/sap/frcnn_keras/result/result-messytable_model_$i/val/log.json 

done

for i in `seq 101 1 200`
do
    cd ~/frcnn_keras/src

    CUDA_VISIBLE_DEVICES=3 python test.py  --save_name messytable --test_path /data1/sap/MessyTable/labels/val.txt --model_idx $i --log_output --load_pickle

    cd ~/frcnn_keras_original

    python script/csv2json.py --src_path /data3/sap/frcnn_keras/result/result-messytable_model_$i/val/log.csv --dst_path /data3/sap/frcnn_keras/result/result-messytable_model_$i/val/log.json 

    python script/json2voc_labels.py --json_path /data3/sap/frcnn_keras/result/result-messytable_model_$i/val/log.json  --simple_label_dir /data3/sap/mAP/input/detection-results

    cd ~/mAP

    python main.py -na -np --output_dir $i
done
END
