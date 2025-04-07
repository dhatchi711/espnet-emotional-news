```bash
git clone https://github.com/Webhose/free-news-datasets.git

cd free-news-datasets/News_Datasets/
# unzip the zip files you are interested, under the same directory

conda create -n sds python=3.9 -y
pip install requirements.txt
```


Step 1: calculate embeddings

```bash
python3 json_title_embeddings.py \
    --dataset_dir "/data/user_data/jiaruil5/11692/free-news-datasets/News_Datasets" \
    --data_base_path /data/user_data/jiaruil5/11692/news_db
```