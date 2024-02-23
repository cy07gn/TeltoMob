
batch_size = args.batch_size
valid_batch_size = args.batch_size
test_batch_size = args.batch_size
data = {}


for category in ['train', 'val', 'test']:

    # Loading npz
    cat_data = np.load(os.path.join(args.data, category + '.npz'))

    if args.log_print:
        print("# Loading:", category + '.npz')
        for k in cat_data.files:
            print(' - col:',k)

    data['x_' + category] = cat_data['x']     # (?, 12, 207, 2)
    data['y_' + category] = cat_data['y']     # (?, 12, 207, 2)

    if args.log_print:
        print(' - x_' +category +':', data['x_' + category].shape)
        print(' - y_' +category +':', data['y_' + category].shape)


# 使用train的mean/std來正規化valid/test #
scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

# 將欲訓練特徵改成正規化
for category in ['train', 'val', 'test']:
    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])


data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
data['scaler'] = scaler

sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adj_data,args.adjtype)   # adjtype: default='doubletransition'

adj_mx = [torch.tensor(i).to(device) for i in adj_mx]

dataloader = data.copy()
