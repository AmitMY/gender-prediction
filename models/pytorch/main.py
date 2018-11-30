import torch
import torch.nn.functional as F

from data.reader import Data
from models.pytorch.load_data import load_dataset
from models.pytorch.models.CNN import CNN
from models.pytorch.models.LSTM import LSTMClassifier
from models.pytorch.models.LSTM_Attn import AttentionModel
from models.pytorch.models.RCNN import RCNN
from models.pytorch.models.RNN import RNN
from models.pytorch.models.selfAttention import SelfAttention

learning_rate = 2e-5
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300


class ModelRunner:
    def __init__(self, model, train, dev, opt={}):

        self.TEXT, self.vocab_size, self.train_iter, self.dev_iter, self.vectors = \
            load_dataset(train, dev, opt)

        self.vectors = self.vectors if "pretrained" in opt else None
        if model == "LSTM":
            self.model = LSTMClassifier(batch_size, output_size, hidden_size, self.vocab_size, embedding_length,
                                        self.vectors)
        elif model == "RNN":
            self.model = RNN(batch_size, output_size, hidden_size, self.vocab_size, embedding_length, self.vectors)
        elif model == "RCNN":
            self.model = RCNN(batch_size, output_size, hidden_size, self.vocab_size, embedding_length, self.vectors)
        elif model == "LSTMAttention":
            self.model = AttentionModel(batch_size, output_size, hidden_size, self.vocab_size, embedding_length,
                                        self.vectors)
        elif model == "SelfAttention":
            self.model = SelfAttention(batch_size, output_size, hidden_size, self.vocab_size, embedding_length,
                                       self.vectors)
        elif model == "CNN":
            self.model = CNN(batch_size, output_size, 1, 128, [3, 4, 5], 1, 0, 0.8, self.vocab_size, embedding_length,
                             self.vectors)
        else:
            raise ValueError("Unknown model " + model)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self):
        def clip_gradient(clip_value):
            params = list(filter(lambda p: p.grad is not None, self.model.parameters()))
            for p in params:
                p.grad.data.clamp_(-clip_value, clip_value)

        def train_model(epoch):
            total_epoch_loss = 0
            total_epoch_acc = 0
            self.model.cuda()
            optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
            steps = 0
            self.model.train()
            for idx, batch in enumerate(self.train_iter):
                text = batch.text[0]
                target = batch.label
                target = torch.autograd.Variable(target).long()
                if torch.cuda.is_available():
                    text = text.cuda()
                    target = target.cuda()
                if (text.size()[
                    0] is not 32):  # One of the batch returned by BucketIterator has length different than 32.
                    continue
                optim.zero_grad()
                prediction = self.model(text)
                loss = loss_fn(prediction, target)
                num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
                acc = 100.0 * num_corrects / len(batch)
                loss.backward()
                clip_gradient(1e-1)
                optim.step()
                steps += 1

                if steps % 100 == 0:
                    print(
                        f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

                total_epoch_loss += loss.item()
                total_epoch_acc += acc.item()

            return total_epoch_loss / len(self.train_iter), total_epoch_acc / len(self.train_iter)

        def eval_model(model, val_iter):
            total_epoch_loss = 0
            total_epoch_acc = 0
            model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(val_iter):
                    text = batch.text[0]
                    if (text.size()[0] is not 32):
                        continue
                    target = batch.label
                    target = torch.autograd.Variable(target).long()
                    if torch.cuda.is_available():
                        text = text.cuda()
                        target = target.cuda()
                    prediction = model(text)
                    loss = loss_fn(prediction, target)
                    num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
                    acc = 100.0 * num_corrects / len(batch)
                    total_epoch_loss += loss.item()
                    total_epoch_acc += acc.item()

            return total_epoch_loss / len(val_iter), total_epoch_acc / len(val_iter)

        loss_fn = F.cross_entropy

        epoch = 0
        while True:
            epoch += 1
            train_loss, train_acc = train_model(epoch)
            val_loss, val_acc = eval_model(self.model, self.dev_iter)
            yield val_acc
            print(
                f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

    # test_loss, test_acc = eval_model(model, test_iter)
    # print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
    #
    # ''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
    # test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
    # test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."
    #
    # test_sen1 = TEXT.preprocess(test_sen1)
    # test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]
    #
    # test_sen2 = TEXT.preprocess(test_sen2)
    # test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]
    #
    # test_sen = np.asarray(test_sen1)
    # test_sen = torch.LongTensor(test_sen)
    # test_tensor = Variable(test_sen, volatile=True)
    # test_tensor = test_tensor.cuda()
    # model.eval()
    # output = model(test_tensor, 1)
    # out = F.softmax(output, 1)
    # if (torch.argmax(out[0]) == 1):
    #     print ("Sentiment: Positive")
    # else:
    #     print ("Sentiment: Negative")


if __name__ == "__main__":
    train, dev = Data("All", "train", ["news"], tokenize=False).split()

    runner = ModelRunner("LSTMAttention", train, dev)
    for d in runner.train():
        print(d)
