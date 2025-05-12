# import faiss
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import pandas as pd
import ast
import itertools
from PIL import Image
from geopy.distance import geodesic
from transformers import CLIPImageProcessor, CLIPModel
from utils.utils import MP16Dataset, im2gps3kDataset, yfcc4kDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

def load_gps_data(csv_file):
    data = pd.read_csv(csv_file)
    lat_lon = data[['LAT', 'LON']]
    gps_tensor = torch.tensor(lat_lon.values, dtype=torch.float32)
    return gps_tensor

class ZeroShotPredictor(nn.Module):
    def __init__(self, model, device='cuda', queue_size=4096):
        super().__init__()

        if isinstance(model, str):
            self.model = torch.load(model, map_location=device)
        else:
            self.model = model
            self.model.to(device)
        self.model.requires_grad_(False)

        self.gps_gallery = load_gps_data("coordinates_100K.csv")
        self._initialize_gps_queue(queue_size)

        self.device = device

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))
    
    # def forward(self, image, location):
    #     image_embeds = self.model.vision_projection_else_2(self.model.vision_projection(self.model.vision_model(image)[1]))
    #     image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True) # b, 768

    #     b, c, _ = location.shape
    #     location = location.reshape(b*c, 2)
    #     location_embeds = self.model.location_encoder(location)
    #     location_embeds = self.model.location_projection_else(location_embeds.reshape(b*c, -1))
    #     location_embeds = location_embeds / location_embeds.norm(p=2, dim=-1, keepdim=True)
    #     location_embeds = location_embeds.reshape(b, c, -1) #  b, c, 768

    #     # image with location
    #     logit_scale = self.model.logit_scale2.exp()
    #     logits_per_image = torch.matmul(image_embeds, location_embeds.t()) * logit_scale

    #     return logits_per_image

    def forward(self, image, location):
        # image: [B, C, H, W]
        # location: [N, 768*3] or [N, 2] depending on where it is used

        # Compute image embeddings
        vision_output = self.model.vision_model(image)[1]  # e.g., [B, 768]
        image_embeds = self.model.vision_projection(vision_output)
        image_embeds = self.model.vision_projection_else_2(image_embeds)  # [B, 768]
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)  # Normalize

        # Process GPS coordinates
        location_embeds = self.model.location_encoder(location)                # [N, F]
        location_embeds = self.model.location_projection_else(location_embeds) # [N, 768]
        location_embeds = location_embeds / location_embeds.norm(p=2, dim=-1, keepdim=True)

        # Compute similarity: [B, 768] x [768, N] -> [B, N]
        logit_scale = self.model.logit_scale2.exp()
        logits_per_image = torch.matmul(image_embeds, location_embeds.t()) * logit_scale

        return logits_per_image  # [B, N]
    

    def predict_image(self, image_path, top_k):
        self.model.eval()

        # Step 1: Load and preprocess the input image
        image = Image.open(image_path).convert("RGB")
        image = self.image_encoder.preprocess_image(image)  # Should return a tensor
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension

        # Step 2: Compute image embedding
        with torch.no_grad():
            vision_output = self.model.vision_model(image)[1]
            
            image_embed = self.model.vision_projection(vision_output)
            image_embed = image_embed / image_embed.norm(p=2, dim=-1, keepdim=True)

            image_text_embed = self.model.vision_projection_else_1(image_embed)
            image_text_embed = image_text_embed / image_text_embed.norm(p=2, dim=-1, keepdim=True)

            image_location_embed = self.model.vision_projection_else_2(image_embed)
            image_location_embed = image_location_embed / image_location_embed.norm(p=2, dim=-1, keepdim=True)

            # Concatenate final embedding
            full_embed = torch.cat([image_embed, image_text_embed, image_location_embed], dim=1)  # Shape: [1, 768*3]

        # Step 3: Compute similarity with GPS gallery
        gps_gallery = self.gps_gallery.to(self.device)  # [N, 768*3]
        logits_per_image = self.forward(full_embed, gps_gallery)  # [1, N]

        # Step 4: Softmax to get probabilities (optional)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Step 5: Top-K predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]  # [top_k, 768*3]
        top_pred_prob = top_pred.values[0]                   # [top_k]

        return top_pred_gps, top_pred_prob
    
    def predict_im2gps3k_dataset(self, top_k):
        self.model.eval()

        # Prepare dataset and dataloader
        dataset = im2gps3kDataset(vision_processor=self.model.vision_processor, text_processor=None, root_path='/kaggle/input/im2gps3k', image_data_path='im2gps3ktest')
        dataloader = DataLoader(
            dataset, batch_size=256, shuffle=False, num_workers=4,
            pin_memory=True, prefetch_factor=5
        )

        gps_gallery = self.gps_gallery.to(self.device)  # [N, 768*3]
        all_topk_gps = []
        all_topk_probs = []

        print('Generating embeddings and computing top-k predictions...')
        for images, _, _, _ in tqdm(dataloader):
            images = images.to(self.device)

            with torch.no_grad():
                # Compute similarity (logits) and top-k
                logits_per_image = self.forward(images, gps_gallery)  # [B, N]
                probs_per_image = logits_per_image.softmax(dim=-1).cpu()  # [B, N]

                top_preds = torch.topk(probs_per_image, top_k, dim=1)  # values: [B, k], indices: [B, k]

                # Store top-k GPS coordinates and probs
                batch_topk_gps = self.gps_gallery[top_preds.indices]  # [B, k, 768*3]
                batch_topk_probs = top_preds.values  # [B, k]

                all_topk_gps.append(batch_topk_gps)
                all_topk_probs.append(batch_topk_probs)

        # Concatenate results for the entire dataset
        all_topk_gps = torch.cat(all_topk_gps, dim=0)     # [Total_Images, k, 768*3]
        all_topk_probs = torch.cat(all_topk_probs, dim=0) # [Total_Images, k]
        print(all_topk_gps)
        return all_topk_gps, all_topk_probs

    def evaluate_im2gps3k(self, df_path, top_k=5):
        """
        Combines prediction and evaluation for im2gps3k dataset using top-k GPS candidates.
        
        Args:
            df_path: Path to CSV containing ground truth columns ['LAT', 'LON'].
            top_k: Number of top predictions to consider.
        
        Returns:
            DataFrame with predicted coordinates and geodesic distances.
        """
        print("Start prediction and evaluation...")
        self.model.eval()

        # Prepare dataset and dataloader
        dataset = im2gps3kDataset(
            vision_processor=self.model.vision_processor,
            text_processor=None,
            root_path='/kaggle/input/im2gps3k',
            image_data_path='im2gps3ktest'
        )
        dataloader = DataLoader(
            dataset, batch_size=256, shuffle=False, num_workers=4,
            pin_memory=True, prefetch_factor=5
        )

        gps_gallery = self.gps_gallery.to(self.device)
        all_topk_gps = []
        all_topk_probs = []

        print("Generating top-k predictions...")
        for images, _, _, _ in tqdm(dataloader):
            images = images.to(self.device)

            with torch.no_grad():
                logits = self.forward(images, gps_gallery)  # [B, N]
                probs = logits.softmax(dim=-1).cpu()         # [B, N]
                top_preds = torch.topk(probs, top_k, dim=1)

                batch_topk_gps = gps_gallery[top_preds.indices]  # [B, k, 768*3]
                batch_topk_probs = top_preds.values              # [B, k]

                all_topk_gps.append(batch_topk_gps)
                all_topk_probs.append(batch_topk_probs)

        all_topk_gps = torch.cat(all_topk_gps, dim=0)     # [Total_Images, k, 768*3]
        all_topk_probs = torch.cat(all_topk_probs, dim=0) # [Total_Images, k]

        # No need to decode coordinates, directly use top-k GPS
        max_indices = torch.argmax(all_topk_probs, dim=1)  # [N]

        final_lat_lons = []
        for i in range(len(max_indices)):
            # Select the GPS coordinates corresponding to the max probability
            lat, lon = all_topk_gps[i, max_indices[i]]
            lat = lat.item()
            lon = lon.item()
            if lat < -90 or lat > 90:
                lat = 0
            if lon < -180 or lon > 180:
                lon = 0
            final_lat_lons.append((lat, lon))

        # Evaluation
        df = pd.read_csv(df_path)
        df['LAT_pred'] = [lat for lat, _ in final_lat_lons]
        df['LON_pred'] = [lon for _, lon in final_lat_lons]
        df['geodesic'] = df.apply(lambda x: geodesic((x['LAT'], x['LON']), (x['LAT_pred'], x['LON_pred'])).km, axis=1)

        print(df.head())
        res = []
        for threshold in [2500, 750, 200, 25, 1]:
            acc = (df['geodesic'] < threshold).mean()
            res.append(acc)
            print(f'{threshold}km level: {acc:.4f}')

        return df, res


def main():
    # Initialize predictor
    predictor = ZeroShotPredictor(
        model_path='g3_5_.pth',  # Path to your model
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Set parameters
    top_k = 5  # Number of top predictions to return
    im2gps3k_path = './data/im2gps3k/im2gps3k_places365.csv'  # Path to im2gps3k dataset

    print("Starting prediction on im2gps3k dataset...")
    # Get predictions and evaluate
    results_df,_ = predictor.evaluate_im2gps3k(
        df_path=im2gps3k_path,
        top_k=top_k
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f'results_im2gps3k_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()