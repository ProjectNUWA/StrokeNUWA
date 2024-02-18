from modelzipper.tutils import *


def filter_data(content):
    # data_with_5_kwds = []
    # data_with_10_kwds = []
    # data_with_15_kwds = []
    # data_with_20_kwds = []
    # data_with_25_kwds = []
    # data_with_30_kwds = []

    # with tqdm(total=len(content)) as pbar:
    #     for i, sample in enumerate(content):
    #         if len(sample['keys']) >= 30:
    #             data_with_30_kwds.append(sample)
    #         elif len(sample['keys']) >= 25:
    #             data_with_25_kwds.append(sample)
    #         elif len(sample['keys']) >= 20:
    #             data_with_20_kwds.append(sample)
    #         elif len(sample['keys']) >= 15:
    #             data_with_15_kwds.append(sample)
    #         elif len(sample['keys']) >= 10:
    #             data_with_10_kwds.append(sample)
    #         elif len(sample['keys']) >= 5:
    #             data_with_5_kwds.append(sample)
    #         pbar.update(1)
            
    # print(f"length of data_with_5_kwds: {len(data_with_5_kwds)}")
    # print(f"length of data_with_10_kwds: {len(data_with_10_kwds)}")
    # print(f"length of data_with_15_kwds: {len(data_with_15_kwds)}")
    # print(f"length of data_with_20_kwds: {len(data_with_20_kwds)}")
    # print(f"length of data_with_25_kwds: {len(data_with_25_kwds)}")
    # print(f"length of data_with_30_kwds: {len(data_with_30_kwds)}")

    data_with_5_paths = []
    data_with_50_paths = []
    data_with_100_paths = []
    data_with_150_paths = []
    data_with_200_paths = []
    data_with_250_paths = []
    data_with_300_paths = []

    with tqdm(total=len(content)) as pbar:
        for i, sample in enumerate(content):
            if len(sample['zs']) >= 300:
                data_with_300_paths.append(sample)
            elif len(sample['zs']) >= 250:
                data_with_250_paths.append(sample)
            elif len(sample['zs']) >= 200:
                data_with_200_paths.append(sample)
            elif len(sample['zs']) >= 150:
                data_with_150_paths.append(sample)
            elif len(sample['zs']) >= 100:
                data_with_100_paths.append(sample)
            elif len(sample['zs']) >= 50:
                data_with_50_paths.append(sample)
            elif len(sample['zs']) >= 5:
                data_with_5_paths.append(sample)
            pbar.update(1)
    
    print(f"length of data_with_5_paths: {len(data_with_5_paths)}")
    print(f"length of data_with_50_paths: {len(data_with_50_paths)}")
    print(f"length of data_with_100_paths: {len(data_with_100_paths)}")
    print(f"length of data_with_150_paths: {len(data_with_150_paths)}")
    print(f"length of data_with_200_paths: {len(data_with_200_paths)}")
    print(f"length of data_with_250_paths: {len(data_with_250_paths)}")
    print(f"length of data_with_300_paths: {len(data_with_300_paths)}")

    final_data = data_with_5_paths + data_with_50_paths + data_with_100_paths + data_with_150_paths + data_with_200_paths
    
    final_data = list(filter(lambda x: len(x['keys']) >= 2, final_data))  # at least two keywords
    return final_data

    # assert NotImplementedError, "not implement yet"


def main(rd):
    # load the data
    file_paths = []
    for i in range(8):
        file_paths.append(os.path.join(rd, f"inference_full_data_compress_1_snaps_{i}.pkl"))
    
    content = [auto_read_data(item) for item in file_paths]
    
    # merge content
    print_c(f"======= merge content =======", "magenta")
    merged_list = [item for sublist in content for item in sublist]
    
    # filter dataset by length
    print_c("======= filter dataset by length =======", "magenta")
    processed_data = filter_data(merged_list)

    # save the data
    print_c(f"======= save content =======", "magenta")
    save_path = os.path.join(rd, f"inference_full_data_compress_1_snaps_merged.pkl")
    b_t = time.time()
    auto_save_data(processed_data, save_path)
    print_c(f"save predictions to {save_path}, total time: {time.time() - b_t}", "magenta")

if __name__ == "__main__":
    fire.Fire(main)
    
    