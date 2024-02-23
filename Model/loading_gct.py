#**fusion**#
batch_size = args.batch_size
valid_batch_size = args.batch_size
test_batch_size = args.batch_size
data_pre = {}

_types = ''

for category in ['train'+_types, 'val'+_types, 'test'+_types]:

    print("# Loading:", category + '.npz')

    # Loading npz
    cat_data = np.load(os.path.join(args.data_pre, category + '.npz'))

    data_pre['x_' + category] = cat_data['x']     # (?, 12, 207, 2)
    data_pre['y_' + category] = cat_data['y']     # (?, 12, 207, 2)

    print(cat_data['x'].shape)
    print('x[0]:',cat_data['x'][0])
    print('y[0]:',cat_data['y'][0])
    print('x[-1]',cat_data['x'][-1])
    print('y[-1]',cat_data['y'][-1])

# 使用train的mean/std來正規化valid/test #
scaler_pre = StandardScaler(mean=data_pre['x_train'+_types][..., 0].mean(), std=data_pre['x_train'+_types][..., 0].std())

# 將欲訓練特徵改成正規化
for category in ['train'+_types, 'val'+_types, 'test'+_types]:
    data_pre['x_' + category][..., 0] = scaler_pre.transform(data_pre['x_' + category][..., 0])


data_pre['train_loader'] = DataLoaderM(data_pre['x_train'+_types], data_pre['y_train'+_types], batch_size)
data_pre['val_loader'] = DataLoaderM(data_pre['x_val'+_types], data_pre['y_val'+_types], valid_batch_size)
data_pre['test_loader'] = DataLoaderM(data_pre['x_test'+_types], data_pre['y_test'+_types], test_batch_size)
data_pre['scaler'] = scaler_pre

sensor_ids_pre, sensor_id_to_ind_pre, adj_mx_pre = load_adj(args.adj_data_pre,args.adjtype)   # adjtype: default='doubletransition'

adj_mx_pre = [torch.tensor(i).to(device) for i in adj_mx_pre]

dataloader_pre = data_pre.copy()
