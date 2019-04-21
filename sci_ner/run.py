from ner import *
from dataloader import *
START_TAG = "<START>"
STOP_TAG = "<STOP>"
boardX_log_dir = './'
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
cpu_num = 10
torch.set_num_threads(cpu_num)
from tensorboardX import SummaryWriter
import argparse
import os
import logging






def train(model_save_dir, logger, source_path, target_path, train_ratio, result_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nerdataset = NerDataset(source_path, target_path, device, train_ratio)
    mydata_loader = torch.utils.data.DataLoader(nerdataset, batch_size=1, shuffle=False, num_workers=10)
    model = BiLSTM_CRF(len(nerdataset.get_word2id()), nerdataset.get_target2id(), EMBEDDING_DIM, HIDDEN_DIM, device)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    epoches = 10
    steps_per_epoch = nerdataset.get_len()
    writer = SummaryWriter(boardX_log_dir)


    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for  n_iter in tqdm.tqdm(range(epoches)):  # again, normally you would NOT do 300 epochs, it is toy data
        batch_step=0
        for sentence, tags in tqdm.tqdm(mydata_loader):
            batch_step += 1
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            sentence_in = sentence.squeeze(0).to(device)
            targets = tags.squeeze(0).to(device)
            # sentence_in = sentence
            # targets = tags
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            # sentence_in = prepare_sequence(sentence, word_to_ix).to(device)
            # targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).to(device)
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets, device).to(device)
            writer.add_scalar('train_loss', loss.item(), n_iter * steps_per_epoch + batch_step)
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
        model.eval()
        nerdataset.set_mode('test')
        with open(result_path, 'w') as f:
            for sentence, tags in tqdm.tqdm(nerdataset):
                sentence = sentence.to(device)
                result = model(sentence, device)
                predict =''.join([w+' ' for w in nerdataset.tagid2ner(result[1])])
                tag = ''.join([w+' ' for w in nerdataset.tagid2ner(tags.numpy().tolist())])
                record = f'predict : {predict}\ntagets : {tag}\nscore :  {result[0].item()}\n'
                f.writelines(record)



        nerdataset.set_mode('train')
        model.train()
        model_name = f'model_{(int(n_iter + 1))}_epochs.pt'
        model_path = (os.path.join(model_save_dir, model_name))
        torch.save(model.state_dict(), model_path)
        logger.info(f'evaluating finish and the model has been saved in ' + model_path)



if __name__ == '__main__':

    target_data = './target.txt'
    source_path = './source.txt'

    parser = argparse.ArgumentParser(description='ner pytorch parameters')
    parser.add_argument('--action','-a',  required=True, choices=['train', 'evaluate', 'predict'])
    parser.add_argument('--source_path', default='./source.txt')
    parser.add_argument('--model_save_dir', default= './model_ckpt')
    parser.add_argument('--target_data', default= './target.txt')
    parser.add_argument('--result_file', default='./result.txt')
    parser.add_argument('--train_ratio', type=float, default=0.85)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    if args.action == "train":
        logger.info(f"start to train the model with train_ratio {args.train_ratio}")
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        train(args.model_save_dir, logger, args.source_path, args.target_data, args.train_ratio, args.result_file)