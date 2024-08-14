import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        return hidden

def train(model, criterion, optimizer, train_data, vocab_size, batch_size, seq_length, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        hidden = model.init_hidden(batch_size)
        for batch_idx in range(0, len(train_data), seq_length):
            inputs, targets = get_batch(train_data, batch_idx, batch_size, seq_length, vocab_size)
            optimizer.zero_grad()
            output, hidden = model(inputs, hidden)
            output = output.view(-1, vocab_size)
            loss = criterion(output, targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 1000 == 0 and batch_idx > 0:
                print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, num_epochs, batch_idx, len(train_data), total_loss / 1000))
                total_loss = 0

def get_batch(data, idx, batch_size, seq_length, vocab_size):
    inputs = torch.zeros(batch_size, seq_length, dtype=torch.long)
    targets = torch.zeros(batch_size, seq_length, dtype=torch.long)
    for i in range(batch_size):
        inputs[i,:] = data[idx + i : idx + i + seq_length]
        targets[i,:] = data[idx + i + 1 : idx + i + seq_length + 1]
    return inputs, targets

def generate_text(model, start_seq, vocab, vocab_to_int, int_to_vocab, max_length=100):
    model.eval()
    hidden = model.init_hidden(1)
    start_seq = start_seq.split()
    for word in start_seq:
        idx = vocab_to_int[word]
        output, hidden = model(torch.tensor([[idx]]), hidden)
    generated_text = start_seq
    for _ in range(max_length):
        output = output.view(-1)
        softmax_scores = F.softmax(output, dim=0).detach().numpy()
        idx = np.random.choice(len(softmax_scores), p=softmax_scores)
        word = int_to_vocab[idx]
        generated_text.append(word)
        if word == '<EOS>':
            break
        output, hidden = model(torch.tensor([[idx]]), hidden)
    return ' '.join(generated_text)

if __name__ == "__main__":
    text = "This is a sample text used for training the language model. It can be replaced with your own dataset."

    tokens = text.split()
    vocab = list(set(tokens))
    vocab_size = len(vocab)
    vocab_to_int = {word: idx for idx, word in enumerate(vocab)}
    int_to_vocab = {idx: word for idx, word in enumerate(vocab)}
    data = np.array([vocab_to_int[word] for word in tokens])
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    batch_size = 32
    seq_length = 20
    num_epochs = 10

    model = LM(vocab_size, embedding_dim, hidden_dim, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, criterion, optimizer, data, vocab_size, batch_size, seq_length, num_epochs)

    start_seq = "This is"
    generated_text = generate_text(model, start_seq, vocab, vocab_to_int, int_to_vocab)
    print("Generated Text:", generated_text)
