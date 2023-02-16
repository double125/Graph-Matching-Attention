python3 visual_100/preprocess_image.py --data trainval
python3 visual_100/preprocess_image.py --data test

mv *.zarr ../VQA/visual_graph/visual_100/
mv *.csv ../VQA/visual_graph/visual_100/

