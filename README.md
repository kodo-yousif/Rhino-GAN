# nose-manipulation

## Running steps

### Step 1

Note: your GPU must support driver 11.7 to run this docker file successfully, Else feel free to adjust the package versions to satisfy your GPU condition

```terminal
docker build -t nose-ai .
```

### Step 2

download this link `https://drive.google.com/file/d/1AT6bNR2ppK8f2ETL_evT27f3R_oyWNHS/view?usp=sharing` into `backend/pretrained_models`

### Step 3

Execute in power shell

Execute this command in the root folder of this project

```powershell
docker run -d --gpus all --privileged -it -p 8000:3000 -p 8001:3001 -v .:/nose-ai --name nose-ai-container nose-ai
```

<b>Now you can view the app at <a href="http://localhost:8000/">http://localhost:8000/</a></b>
<b>Note: be careful initial load and requests are heavy due to model loads</b>

## to run the servers manually (Optional)

```terminal
docker exec -it nose-ai-container bash

/opt/conda/bin/conda init bash

echo -e "source /opt/conda/etc/profile.d/conda.sh\nconda activate nose-ai" >> ~/.bashrc

exit
```

if conda env nose-ai is not activates run this

```terminal
conda activate nose-ai
```

**To run the front end server**

```terminal
cd frontend

yarn

yarn dev
```

**To run the backend end server**

```terminal
conda activate nose-ai

python backend/main.py
```

## to run the servers manually (Optional)

This script is to data samples or manually download <a href="https://drive.google.com/drive/folders/15jsR9yy_pfDHiS9aE3HcYDgwtBbAneId">https://drive.google.com/drive/folders/15jsR9yy_pfDHiS9aE3HcYDgwtBbAneId</a> content into `backend/images/unprocessed`

```terminal
python backend/images/unprocessed/populate.py
```
