python3 visual_36/preprocess_image.py --data trainval
python3 visual_36/preprocess_image.py --data test

mv *.zarr ../VQA/visual_graph/vg_36/
mv *.csv ../VQA/visual_graph/vg_36/

