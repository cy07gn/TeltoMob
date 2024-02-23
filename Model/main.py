
def main(runid):

    # if args.load_static_feature:
    #     static_feat = load_node_feature('data/sensor_graph/location.csv')
    # else:
    #     static_feat = None

    model = gginet(args.model_type,
                   args.num_nodes,
                   device,
                   predefined_A=adj_mx,

                   dropout=args.dropout,  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential, conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, layer_norm_affline=True)
    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len,
                     data['scaler'], device, args.cl)


    #**fusion**#
    # 用mobility scaler試試看
    model_pre = gginet(args.model_type,

                   args.num_nodes_pre,
                   device,
                   predefined_A=adj_mx_pre,

                   dropout=args.dropout, node_dim=args.node_dim, dilation_exponential=args.dilation_exponential, conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, layer_norm_affline=True)

    engine_pre = Trainer_pretrained(
                      model_pre,
                      args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len,
                      data['scaler'], device, args.cl)


    #**fusion**#
    # 載入model
    SAVE_PATH = args.save + "exp" + str(args.expid_pre) + "_3.pth"
    print("### loading model is:",SAVE_PATH ,'###')
    checkpoint = torch.load(SAVE_PATH)
    engine_pre.model.load_state_dict(checkpoint['model_state_dict'])
    engine_pre.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    minl = 1e5
    start_epoch=0
    SAVE_PATH = ""
    train_loss_epoch = []  # 紀錄train在epoch收斂
    valid_loss_epoch = []  # 紀錄valid在epoch收斂

    for i in range(start_epoch,start_epoch+args.epochs+1):

        train_loss = []
        train_mape = []
        train_rmse = []
        train_smape = []
        t1 = time.time()
        #dataloader['train_loader'].shuffle()  # 為了檢視資料先拿掉
        #**fusion**#
        permutation = np.random.permutation(dataloader['train_loader'].size)
        dataloader['train_loader'].set_permutation(permutation)
        dataloader_pre['train_loader'].set_permutation(permutation)

        #for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        #**fusion**#
        for iter, ((x, y), (x2, y2)) in enumerate(zip(dataloader['train_loader'].get_iterator(), dataloader_pre['train_loader'].get_iterator())):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)

            #**fusion**#
            trainx2 = torch.Tensor(x2).to(device)
            trainx2= trainx2.transpose(1, 3)
            trainy2 = torch.Tensor(y2).to(device)
            trainy2 = trainy2.transpose(1, 3)

            #print('x2', x2.shape)
            #**fusion**#
            # 要輸入完整trainy，因為要用到time
            pre_trained_output = engine_pre.eval(trainx2,trainy2)
            trainx = pre_trained_output

            metrics = engine.train(trainx, trainy[:,0,:,:])

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_smape.append(metrics[3])

            #sys.exit()

            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        valid_smape = []

        s1 = time.time()
        #for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        #**fusion**#
        for iter, ((x, y), (x2, y2)) in enumerate(zip(dataloader['val_loader'].get_iterator(), dataloader_pre['val_loader'].get_iterator())):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)

            #**fusion**#
            testx2 = torch.Tensor(x2).to(device)
            testx2= testx2.transpose(1, 3)
            testy2 = torch.Tensor(y2).to(device)
            testy2 = testy2.transpose(1, 3)

            #**fusion**#
            # 要輸入完整trainy，因為要用到time
            pre_trained_output = engine_pre.eval(testx2,testy2)
            testx = pre_trained_output


            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_smape.append(metrics[3])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_smape = np.mean(train_smape)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_smape = np.mean(valid_smape)
        #his_loss.append(mvalid_loss)
        his_loss.append(mvalid_smape)

        #writer.add_scalar("train_loss", mtrain_loss, i)
        #writer.add_scalar("valid_loss", mvalid_loss, i)

        writer.add_scalar("train_loss", mvalid_loss, i)
        writer.add_scalar("valid_loss", mvalid_loss, i)


        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        # 紀錄每個epoch的loss
        train_loss_epoch.append(mtrain_loss)
        valid_loss_epoch.append(mvalid_loss)

        '''
        if mvalid_loss<minl:
            torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth")
            minl = mvalid_loss
        '''
        if mvalid_loss<minl:
            target_best_model = args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"
            print("### Update Best Model:",target_best_model, 'Loss:', mvalid_mape, " ###")
            #torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth")
            SAVE_PATH = args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"
            torch.save({
              'epoch': i,
              'task_level': engine.task_level,
              'model_state_dict': engine.model.state_dict(),
              'optimizer_state_dict': engine.optimizer.state_dict(),
              'loss': mvalid_mape,
              'train_loss': train_loss_epoch,
              'valid_loss': valid_loss_epoch
            }, SAVE_PATH)
            minl = mvalid_loss

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    bestid = np.argmin(his_loss)

    writer.close()
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))

    #target_model = args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"
    SAVE_PATH = args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"
    print("### loading model is:",SAVE_PATH ,'###')
    #engine.model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"))
    checkpoint = torch.load(SAVE_PATH)
    engine.model.load_state_dict(checkpoint['model_state_dict'])
    engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    print("### Loading Model finished ###")
    print("### The valid loss on loding model is", str(round(loss,4)))

    # 只更新最後的train loss
    #checkpoint['train_loss'] = train_loss_epoch
    #checkpoint['valid_loss'] = valid_loss_epoch
    torch.save({
      'epoch': checkpoint['epoch'],  # best epoch
      'task_level': checkpoint['task_level'],
      'model_state_dict': checkpoint['model_state_dict'],
      'optimizer_state_dict': checkpoint['optimizer_state_dict'],
      'loss': checkpoint['loss'],
      'train_loss': checkpoint['train_loss'],
      'valid_loss': checkpoint['valid_loss']
    }, SAVE_PATH)
    ### 測試讀取出的model ###
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    valid_smape = []
    s1 = time.time()
    #for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
    #**fusion**#
    for iter, ((x, y), (x2, y2)) in enumerate(zip(dataloader['val_loader'].get_iterator(), dataloader_pre['val_loader'].get_iterator())):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)

        #**fusion**#
        testx2 = torch.Tensor(x2).to(device)
        testx2= testx2.transpose(1, 3)
        testy2 = torch.Tensor(y2).to(device)
        testy2 = testy2.transpose(1, 3)

        #**fusion**#
        # 要輸入完整trainy，因為要用到time
        pre_trained_output = engine_pre.eval(testx2,testy2)
        testx = pre_trained_output

        metrics = engine.eval(testx, testy[:,0,:,:])
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
        valid_smape.append(metrics[3])

    mvalid_loss = np.mean(valid_loss)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    print("### 2-The valid loss on loding model is", str(round(mvalid_mape,4)))
    ### 測試讀取出的model ###

    #valid data
    outputs = []
    realy = torch.Tensor(dataloader['y_val']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]
    print('#realy', realy.shape)

    #for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
    #**fusion**#
    for iter, ((x, y), (x2, y2)) in enumerate(zip(dataloader['val_loader'].get_iterator(), dataloader_pre['val_loader'].get_iterator())):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)

        #**fusion**#
        testx2 = torch.Tensor(x2).to(device)
        testx2= testx2.transpose(1, 3)
        testy2 = torch.Tensor(y2).to(device)
        testy2 = testy2.transpose(1, 3)

        #**fusion**#
        # 要輸入完整trainy，因為要用到time
        pre_trained_output = engine_pre.eval(testx2,testy2)
        testx = pre_trained_output


        #print('testx2', testx.shape)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1,3)  # 64,1,6,12

        outputs.append(preds.squeeze()) # 64,1,6,12 ->squeeze()->64,6,12

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]  # 5240,6,12
    print('# cat valid preds', yhat.shape)

    pred = data['scaler'].inverse_transform(yhat)

    vmae, vmape, vrmse,vsmape = metric(pred,realy)
    print("valid mape",vmape)

    #test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    #for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    #**fusion**#
    for iter, ((x, y), (x2, y2)) in enumerate(zip(dataloader['test_loader'].get_iterator(), dataloader_pre['test_loader'].get_iterator())):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)


        #**fusion**#
        testx2 = torch.Tensor(x2).to(device)
        testx2= testx2.transpose(1, 3)
        testy2 = torch.Tensor(y2).to(device)
        testy2 = testy2.transpose(1, 3)

        #**fusion**#
        # 要輸入完整trainy，因為要用到time
        pre_trained_output = engine_pre.eval(testx2,testy2)
        testx = pre_trained_output


        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]  #10478, 6, 12
    print('# cat test preds', yhat.shape)

    mae = []
    mape = []
    rmse = []
    smape = []
    for i in range(args.seq_out_len):

        pred = data['scaler'].inverse_transform(yhat[:, :, i])

        real = realy[:, :, i]

        metrics = metric(pred, real)

        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
        smape.append(metrics[3])

    log = '{:.2f}	{:.2f}	{:.4f}	{:.4f}	'
    print( "##### exp" + str(args.expid) + "_" + str(runid)+'	',
          log.format(mae[0], rmse[0], smape[0], mape[0]),
          log.format(mae[1], rmse[1], smape[1], mape[1]),
          log.format(mae[2], rmse[2], smape[2], mape[2]),
          log.format(mae[3], rmse[3], smape[3], mape[3]),
         )

    ### Drawing Loss Diagram ###
    fig = plt.figure(figsize=(10, 6), dpi=600)
    plt.plot(checkpoint['train_loss'], label="train loss")
    plt.plot(checkpoint['valid_loss'], label="valid loss")
    plt.legend(loc="upper right")
    plt.title('#Loss of Training', fontsize=20)
    plt.ylabel("MAPE", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.show()

    return vmae, vmape, vrmse,vsmape, mae, mape, rmse,smape

if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    vsmape = []
    mae = []
    mape = []
    rmse = []
    smape = []
    for i in range(args.runs):
        vm1, vm2, vm3,vm4, m1, m2, m3, m4 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        vsmape.append(vm4)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)
        smape.append(m4)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)
    smape = np.array(smape)

    amae = np.mean(mae,0)
    amape = np.mean(mape,0)
    armse = np.mean(rmse,0)
    asmape = np.mean(smape,0)

    smae = np.std(mae,0)
    s_mape = np.std(mape,0)
    srmse = np.std(rmse,0)
    s_smape = np.std(smape,0)

    print('\n\nResults for 10 runs\n\n')
    #valid data
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae),np.mean(vrmse),np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae),np.std(vrmse),np.std(vmape)))
    print('\n\n')
    #test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in [0,1,2,3]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i+1, amae[i], armse[i], amape[i], smae[i], srmse[i], s_mape[i]))
