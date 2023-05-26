from firebase_admin import initialize_app, storage
from pkg.config.config import cfg
from pkg.logger.logger import logger

if cfg.get("ENV") == "prod":
    PROJECT_ID = "prod"
else:
    PROJECT_ID = "dev"

initialize_app(
    options={"storageBucket": PROJECT_ID + ".appspot.com"}
)

bucket = storage.bucket()

def download_file_from_bucket(local_path: str, bucket_path: str):
    logger.info(f'Start downloading file from {bucket_path} to {local_path}')
    bucket.blob(bucket_path).download_to_filename(local_path)
    logger.info(f'File downloaded to {local_path}')
