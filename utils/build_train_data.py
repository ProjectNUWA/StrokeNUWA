import os
import glob
import logging
import json
import sys
from tqdm import tqdm
from concurrent import futures
from argparse import ArgumentParser
from change_deepsvg.svglib.svg import SVG
from change_deepsvg.svglib.geom import Bbox, Angle, Point
from change_deepsvg.difflib.tensor import SVGTensor
from modelzipper.tutils import *
import torch
from tqdm import trange
from tqdm.auto import tqdm 
import multiprocessing

BLACK_BOX = torch.tensor([
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  96.],
    [  1.,   0.,  96.,   0.,   0.,   0.,   0.,   0., 191.],
    [  1.,   0., 191.,   0.,   0.,   0.,   0.,  96., 191.],
    [  1.,  96., 191.,   0.,   0.,   0.,   0., 191., 191.],
    [  1., 191., 191.,   0.,   0.,   0.,   0., 191.,  96.],
    [  1., 191.,  96.,   0.,   0.,   0.,   0., 191.,   0.],
])

EDGE = torch.tensor([
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   4., 104.],
    [  1.,   4., 104.,   0.,   0.,   0.,   0.,   4., 199.],
    [  1.,   4., 199.,   0.,   0.,   0.,   0., 199., 199.],
    [  1., 199., 199.,   0.,   0.,   0.,   0., 199.,   4.],
    [  1., 199.,   4.,   0.,   0.,   0.,   0.,   4.,   4.],
    [  1.,   4.,   4.,   0.,   0.,   0.,   0.,   4., 104.],
    [  1.,   4., 104.,   0.,   0.,   0.,   0.,   4., 104.],
])

def build_mesh_data(svg_file):
    try:
        svg = SVG.load_svg(svg_file)
        svg.numericalize(n=200)
        svg_tensors = svg.to_tensor(concat_groups=False, PAD_VAL=0)
        svg_tensors = torch.cat(svg_tensors)
        if svg_tensors[:6].equal(BLACK_BOX):
            svg_tensors = svg_tensors[6:]
        if svg_tensors[:7].equal(EDGE):
            svg_tensors = svg_tensors[7:]
        svg_tensors = torch.clamp(svg_tensors, min=0, max=200)
        return svg_tensors
    except:
        print_c("Error in build_mesh_data", 'red')
        return None

def convert_to_mesh(mesh_data, num_sub_path = 3):
    idx = 0
    pair_sub_paths = []
    new_mesh_data = mesh_data.copy()
    for i in range(0, len(mesh_data), num_sub_path):
        if i + num_sub_path <= len(mesh_data):
            pair_sub_paths.append([k for k in range(idx, idx + num_sub_path)])
            idx += num_sub_path
        elif i < len(mesh_data) and len(mesh_data) - i < num_sub_path:
            pair_sub_paths.append([k for k in range(idx, idx + num_sub_path)])
            new_mesh_data = torch.cat([new_mesh_data, torch.zeros(num_sub_path - (len(mesh_data) - i), 9)])
        else:
            break
    return new_mesh_data, pair_sub_paths

def convert_svg(sample):
    svg_file = sample['file_path']
    keywords = sample['keywords']
    category_name = sample['category_name']
    mesh_data = build_mesh_data(svg_file)
    return {'keywords': keywords, 'mesh_data': mesh_data, 'category_name': category_name}


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--workers", default=32, type=int)
    parser.add_argument("--file", default=32, type=int)

    args = parser.parse_args()

    meta_data = auto_read_data(args.file)  # ["file_path", "keywords", "category_name"]


    auto_save_data(meta_data, '/save/path')