import boto3
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sklearn.neighbors import NearestNeighbors


#/ Read from CSV
# Sample user-item interaction data (Replace this with your real data)
data = pd.read_csv('data3.csv')
df = pd.DataFrame(data)
# Create a user-item interaction matrix
interaction_matrix = df.pivot(index='user_id', columns='topic_id', values='rating').fillna(0)

# Convert the interaction matrix to a NumPy array
interaction_matrix_array = interaction_matrix.values

#/ Build the KNN model
k = 3  # Number of neighbors to consider (you can adjust this)
knn_model = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
knn_model.fit(interaction_matrix_array)

# Define a function to get recommendations for a target user
async def _get_recommendations(target_user: int, k: int) -> list[int]:
    target_user_idx = interaction_matrix.index.get_loc(target_user)
    _, indices = knn_model.kneighbors(interaction_matrix_array[target_user_idx].reshape(1, -1))
    similar_user_indices = indices[0]
    similar_users = interaction_matrix.index[similar_user_indices]
    recommendations = []
    for user in similar_users:
        user_interactions = interaction_matrix.loc[user]
        recommendations.extend(user_interactions[user_interactions == 0].index.tolist())
        if len(recommendations) >= k:
            break
    return recommendations[:k]

# FastAPI app
app = FastAPI()

# Config CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TODO: Change this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to S3 using Boto3
s3 = boto3.client(
    's3',
    aws_access_key_id='AKIAR4DRRN52DUHTP7NZ',
    aws_secret_access_key='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'
)
bucket_name = 'noy-final-project'


### ALL ENDPOINTS ###

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Hello World"}


# Recommendation endpoint
@app.get("/recommendations")
async def get_recommendations(target_user: int, k: int = 5):
    print(f"Getting recommendations for user {target_user}... (k={k})")
    try:
        return await _get_recommendations(target_user, k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Upload endpoint
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Check file type is image
    if not file.filename.endswith(".jpg") and not file.filename.endswith(".jpeg") and not file.filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="File type not supported, only jpg, jpeg and png files are supported")

    # Upload the file to S3 bucket
    try:
        s3.upload_fileobj(file.file, bucket_name, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"message": "File uploaded successfully"}


# Download endpoint
@app.get("/download/{file_name}")
async def download_file(file_name: str):
    # If file_name is empty, return a list of files in the bucket
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_name)
        # Get the file contents
        file_contents = response['Body'].read()

        # Check file_content type is image
        content_type = ""
        if file_name.endswith(".jpg"):
            content_type = "image/jpg"
        elif file_name.endswith(".jpeg"):
            content_type = "image/jpeg"
        elif file_name.endswith(".png"):
            content_type = "image/png"
        else:
            # Else file content is octet-stream
            content_type = "application/octet-stream"

        # Return the file contents and a response header
        return Response(content=file_contents, media_type=content_type, headers={"Content-Disposition": "attachment; filename=" + file_name})

    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
