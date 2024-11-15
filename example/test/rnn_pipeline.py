import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import collections
import matplotlib.pyplot as plt
import gensim.downloader as api
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
import tqdm

class ModelType:
    RNN = 'RNN'
    LSTM = 'LSTM'
    RNN_MaxPool = 'RNN_MaxPool'

class TextClassifierRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, pretrained_embedding):
        super(TextClassifierRNN, self).__init__()
        # use pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        
        # Pack the embedded sequences
        packed_embedded = rnn_utils.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
  
        packed_output, _ = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # get last hidden state with list slice
        output = output[:, -1, :]
        
        # sigmoid
        output = self.fc(output)
        # pass to sigmoid
        sig_out = self.sigmoid(output)
        sig_out = sig_out.squeeze(1)
        return sig_out

class TextClassifierLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pretrained_embedding):
        super(TextClassifierLSTM, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        
        # Pack the embedded sequences
        packed_embedded = rnn_utils.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        
        # Use the last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        output = self.fc(hidden)
        sig_out = self.sigmoid(output)
        sig_out = sig_out.squeeze(1)
        
        return sig_out
    
class TextClassifierRNNMaxPool(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, pretrained_embedding):
        super(TextClassifierRNNMaxPool, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, lengths):
        embedded = self.embedding(x)
        
        # Pack the embedded sequences
        packed_embedded = rnn_utils.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        # Pass through the RNN
        packed_output, hidden = self.rnn(packed_embedded)
        
        packed_output, _ = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # max pool on output
        output, _ = torch.max(output, dim=1)
        
        # sigmoid
        output = self.fc(output)
        # pass to sigmoid
        sig_out = self.sigmoid(output)
        sig_out = sig_out.squeeze(1)
        return sig_out
    
class TextClassificationPipeline:
    def __init__(self, augmented_data, test_data, max_length, test_size, batch_size=512, 
                 embedding_dim=300, hidden_dim=300, n_layers=2, bidirectional=True, dropout_rate=0.5, lr=5e-4, model_type=ModelType.RNN_MaxPool):
        
        # Set initial attributes
        self.augmented_data = augmented_data
        self.test_data = test_data
        self.max_length = max_length
        self.test_size = test_size
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.model_type = model_type
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prepare data and model
        self._prepare_data()
        self._build_model()

    def tokenize_example(self, example, max_length):
        tokens = word_tokenize(example["text"])[:max_length]
        length = len(tokens)
        return {"tokens": tokens, "length": length}
    
    def build_vocab(self, data, min_count=5):
        """Builds a vocabulary from a list of sentences.

        Args:
            data: A list of sentences.
            min_count: The minimum frequency for a word to be included in the vocabulary.

        Returns:
            A gensim Dictionary object.
        """

        dictionary = Dictionary(data)
        dictionary.filter_extremes(no_below=min_count, no_above=1.0)
        dictionary.add_documents([["<unk>", "<pad>"]])

        return dictionary

    def numericalize_example(self, example, vocab):
        doc_bow = vocab.doc2bow(example["tokens"])
        ids = [id for id, _ in doc_bow]
        return {"ids": ids}
    
    def get_collate_fn(self, pad_index):
        def collate_fn(batch):
            batch_ids = [i["ids"] for i in batch]
            batch_ids = nn.utils.rnn.pad_sequence(
                batch_ids, padding_value=pad_index, batch_first=True
            )
            batch_length = [i["length"] for i in batch]
            batch_length = torch.stack(batch_length)
            batch_label = [i["label"] for i in batch]
            batch_label = torch.stack(batch_label)
            batch = {"ids": batch_ids, "length": batch_length, "label": batch_label}
            return batch

        return collate_fn

    def get_data_loader(self, dataset, batch_size, pad_index, shuffle=False):
        collate_fn = self.get_collate_fn(pad_index)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
        )
        return data_loader

    def train(self, dataloader, model, criterion, optimizer, device='cpu'):
        if self.device:
            device = self.device
        model.train()
        epoch_losses = []
        epoch_accs = []
        for batch in tqdm.tqdm(dataloader, desc="training..."):
            ids = batch["ids"].to(device)
            length = batch["length"]
            length = torch.clamp(length, max=ids.size(1))  
            label = batch["label"].to(device)

            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = self.get_accuracy(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
        return np.mean(epoch_losses), np.mean(epoch_accs)

    def evaluate(self, dataloader, model, criterion, device='cpu'):
        if self.device:
            device = self.device
        model.eval()
        epoch_losses = []
        epoch_accs = []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="evaluating..."):
                ids = batch["ids"].to(device)
                length = batch["length"]
                length = torch.clamp(length, max=ids.size(1))
                label = batch["label"].to(device)
                prediction = model(ids, length)
                loss = criterion(prediction, label)
                accuracy = self.get_accuracy(prediction, label)
                epoch_losses.append(loss.item())
                epoch_accs.append(accuracy.item())
        return np.mean(epoch_losses), np.mean(epoch_accs)

    def get_accuracy(self, prediction, label):
        batch_size, _ = prediction.shape
        predicted_classes = prediction.argmax(dim=-1)
        correct_predictions = predicted_classes.eq(label).sum()
        accuracy = correct_predictions / batch_size
        return accuracy

    def _prepare_data(self):
        # Tokenize augmented dataset
        self.augmented_data = self.augmented_data.map(self.tokenize_example, fn_kwargs={"max_length": self.max_length})
        self.test_data = self.test_data.map(self.tokenize_example, fn_kwargs={"max_length": self.max_length})

        # Split train and validation data
        train_valid_split = self.augmented_data.train_test_split(test_size=self.test_size)
        self.train_data = train_valid_split["train"]
        self.valid_data = train_valid_split["test"]

        # Build vocabulary
        min_freq = 5
        self.vocab = self.build_vocab(self.train_data["tokens"], min_freq)
        self.vocab.compactify()
        unk_index = self.vocab.token2id.get("<unk>")
        pad_index = self.vocab.token2id.get("<pad>")
        self.vocab.default_index = unk_index

        # Numericalize datasets
        self.train_data = self.train_data.map(self.numericalize_example, fn_kwargs={"vocab": self.vocab})
        self.valid_data = self.valid_data.map(self.numericalize_example, fn_kwargs={"vocab": self.vocab})
        self.test_data = self.test_data.map(self.numericalize_example, fn_kwargs={"vocab": self.vocab})

        # Set data format for PyTorch
        self.train_data = self.train_data.with_format(type="torch", columns=["ids", "label", "length"])
        self.valid_data = self.valid_data.with_format(type="torch", columns=["ids", "label", "length"])
        self.test_data = self.test_data.with_format(type="torch", columns=["ids", "label", "length"])

        # Create data loaders
        self.train_data_loader = self.get_data_loader(self.train_data, self.batch_size, pad_index, shuffle=True)
        self.valid_data_loader = self.get_data_loader(self.valid_data, self.batch_size, pad_index)
        self.test_data_loader = self.get_data_loader(self.test_data, self.batch_size, pad_index)

    def _build_model(self):
        # Initialize model
        vocab_size = len(self.vocab)
        output_dim = len(self.train_data.unique("label"))

        # Initialize word embeddings
        word_vectors = api.load('glove-wiki-gigaword-300')
        words_in_vocab = list(self.vocab.token2id.keys())
        pretrained_embedding = torch.zeros(len(self.vocab), word_vectors.vector_size).to(self.device)

        for i, word in enumerate(words_in_vocab):
            if word in word_vectors:
                pretrained_embedding[i] = torch.tensor(word_vectors[word])
        
        if(self.model_type == ModelType.RNN_MaxPool):
            self.model = TextClassifierRNNMaxPool(
            vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            output_dim,
            self.n_layers,
            pretrained_embedding
            ).to(self.device)
        elif(self.model_type== ModelType.LSTM):
            self.model = TextClassifierLSTM(
            vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            output_dim,
            self.n_layers,
            self.bidirectional,
            self.dropout_rate,
            pretrained_embedding
            ).to(self.device)
        elif(self.model_type == ModelType.RNN):
            self.model = TextClassifierRNN(
            vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            output_dim,
            self.n_layers,
            pretrained_embedding
            ).to(self.device)
        else:
            raise Exception('Invalid ModelType')

        # Set optimizer and loss criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self, n_epochs=10):
        self.metrics = collections.defaultdict(list)
        best_valid_loss = float("inf")

        for epoch in range(n_epochs):
            train_loss, train_acc = self.train(self.train_data_loader, self.model, self.criterion, self.optimizer, self.device)
            valid_loss, valid_acc = self.evaluate(self.valid_data_loader, self.model, self.criterion, self.device)

            self.metrics["train_losses"].append(train_loss)
            self.metrics["train_accs"].append(train_acc)
            self.metrics["valid_losses"].append(valid_loss)
            self.metrics["valid_accs"].append(valid_acc)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), "best_model.pt")

            print(f"Epoch: {epoch + 1}/{n_epochs}")
            print(f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}")
            print(f"Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.3f}")

    def plot_loss(self, title=''):
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.plot(self.metrics["train_losses"], label="Train Loss")
        plt.plot(self.metrics["valid_losses"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(range(len(self.metrics["train_losses"])))
        plt.legend()
        plt.grid()
        plt.show()

    def plot_acc(self, title=''):
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.plot(self.metrics["train_accs"], label="Train Acc")
        plt.plot(self.metrics["valid_accs"], label="Validation Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xticks(range(len(self.metrics["train_accs"])))
        plt.legend()
        plt.grid()
        plt.show()