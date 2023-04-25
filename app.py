from flask import Flask, render_template, request
import torch
import torchvision.transforms as T
import torchvision.transforms as transforms
import torch.nn as nn  # Neural Network
import torch.nn.functional as F
import torchvision.models as models 
import pandas as pd
from collections import Counter
import spacy
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import torchvision.models as models 
import io
# from predict import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = './model_final.pt'
my_vocab = './vocab.pt'
data_location = 'dataset'


app = Flask(__name__)

# model.load_state_dict(torch.load('path_to_saved_model.pth', map_location=torch.device('cpu')))
transforms = T.Compose([
    T.Resize(226),        # Resizing the images              
    T.RandomCrop(224),    # Random croping for the image - Data Augmentation            
    T.ToTensor(),         # Converting into Tensor -         
    # T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))  # Normalizing the image ternsor - for faster training
])

caption_file = data_location + '/captions.txt'  # locating the file
df = pd.read_csv(caption_file)  # reading captions file

spacy_eng = spacy.load('en_core_web_sm')


class Vocabulary:

    # Define the constructor method for the class
    def __init__(self,freq_threshold):
        
        # Initialize a dictionary called itos with special tokens at specific indices
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        
        # Initialize a dictionary called stoi with keys and values of itos inverted
        self.stoi = {v:k for k,v in self.itos.items()}
        
        # Initialize a dictionary called stoi with keys and values of itos inverted
        self.freq_threshold = freq_threshold
        
    # Define a method to get the length of the vocabulary
    def __len__(self): return len(self.itos)
    
    # Define a static method to tokenize text using spaCy tokenizer
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]
    
    # Define a method to build the vocabulary from a list of sentences
    def build_vocab(self, sentence_list):

        # Initialize a counter to keep track of word frequencies
        frequencies = Counter()
        idx = 4
        
        # Loop through each sentence in the list of sentences
        for sentence in sentence_list:

            # Tokenize the sentence using the tokenize method defined earlier
            for word in self.tokenize(sentence):
                # Increment the frequency of the current word
                frequencies[word] += 1
                
                # If the frequency of the current word reaches the frequency threshold,
                # add it to the vocabulary and increment the index
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    # Define a method to convert a text into a list of numerical tokens based on the vocabular
    def numericalize(self,text):
      
        # Tokenize the text using the tokenize method defined earlier
        tokenized_text = self.tokenize(text)

        # Convert each token into its corresponding index in the vocabulary (or <UNK> if it's not in the vocabulary)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]   
class FlickrDataset(Dataset):
    """
    FlickrDataset
    """
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5):

        # Initialize dataset attributes
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform
        
        # Extract image names and captions from dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        # Create vocabulary object and build vocab from captions
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
        
    # Return the length of the dataset
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        # Retrieve the caption and image name for the given index
        caption = self.captions[idx]
        img_name = self.imgs[idx]

        # Construct the path to the image file and load it
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")
        
        # Apply any specified image transforms
        if self.transform is not None:
            img = self.transform(img)
        
        # Convert the caption to a numericalized vector using the vocabulary
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]] # Start-of-sentence token
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]] # End-of-sentence token
        
        # Return the image tensor and caption tensor as a tuple
        return img, torch.tensor(caption_vec)
dataset =  FlickrDataset(
    root_dir = data_location+"/Images", # Images Data Location
    captions_file = data_location+"/captions.txt",  # Captions data location
    transform=transforms  # Applying transfromations
)
# Defining the EncoderCNN class as a subclass of nn.Module
class EncoderCNN(nn.Module):
    def __init__(self):
        # Initializing the superclass(nn.Module)
        super(EncoderCNN, self).__init__()
        
        # Initializing a pre-trained ResNet50 model
        resnet = models.resnet50(pretrained=True)
        
        # Freezing the weights of the ResNet50 model
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # Extracting all layers of the ResNet50 model except the last two layers
        modules = list(resnet.children())[:-2]
        
        # Defining a Sequential model with the extracted modules
        self.resnet = nn.Sequential(*modules)
        
    # Defining the forward pass of the EncoderCNN class
    def forward(self, images):
        # Forward propagating the images through the ResNet50 model
        features = self.resnet(images)
        
        # Permuting the dimensions of the features tensor
        features = features.permute(0, 2, 3, 1)
        
        # Flattening the features tensor
        features = features.view(features.size(0), -1, features.size(-1))
        
        # Returning the final features tensor
        return features
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        
        # initialize attention layer parameters
        self.attention_dim = attention_dim
        self.W = nn.Linear(decoder_dim, attention_dim) # linear layer to transform decoder hidden state
        self.U = nn.Linear(encoder_dim, attention_dim) # linear layer to transform encoder features
        self.A = nn.Linear(attention_dim, 1)           # linear layer to get attention scores
        
    def forward(self, features, hidden_state):
        # transform the encoder features and decoder hidden state
        u_hs = self.U(features)    # shape: (batch_size, num_pixels, attention_dim)
        w_ah = self.W(hidden_state) # shape: (batch_size, attention_dim)
        
        # calculate attention scores and weights
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) # shape: (batch_size, num_pixels, attention_dim)
        attention_scores = self.A(combined_states) # shape: (batch_size, num_pixels, 1)
        attention_scores = attention_scores.squeeze(2) # shape: (batch_size, num_pixels)
        
        # normalize the attention scores across the image pixels
        alpha = F.softmax(attention_scores,dim=1) # shape: (batch_size, num_pixels)
        
        # calculate the weighted average of the encoder features using the attention weights
        attention_weights = features * alpha.unsqueeze(2) # shape: (batch_size, num_pixels, encoder_dim)
        attention_weights = attention_weights.sum(dim=1)  # shape: (batch_size, encoder_dim)
        
        # return the attention scores and the weighted average of the encoder features
        return alpha, attention_weights
# Attention Decoder module to generate captions
class DecoderRNN(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        
        #save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        # generate embeddings for words
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        
        # features representation
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim,decoder_dim,bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        
        # linear layer that outputs one-hot vector of predicted word
        self.fcn = nn.Linear(decoder_dim,vocab_size)
        self.drop = nn.Dropout(drop_prob)
        
    # inputs are feature representation and captions (vectors)    
    def forward(self, features, captions):
        
        #vectorize the caption
        embeds = self.embedding(captions)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        #get the seq length to iterate
        seq_length = len(captions[0])-1 #Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)
        
        # predicted captions in form of one-hot vectors
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length,num_features).to(device)
        
        # feed in the input for each time instance along with context vectors
        for s in range(seq_length):
            # first, pass the features and decoder hidden state
            alpha,context = self.attention(features, h)
            
            # lstm input are embeddings repr words and context vectors
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            
            # hidden state for next time instance
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            # pass through dropout layer
            output = self.fcn(self.drop(h))
            
            # get the prediction and weights
            preds[:,s] = output
            alphas[:,s] = alpha  
        
        return preds, alphas
    
    def generate_caption(self,features,max_len=50,vocab=None):
        # Inference part
        # Given the image features generate the captions
        
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        # store the weights
        alphas = []
        
        #starting input
        word = torch.tensor(vocab.stoi['<SOS>']).view(1,-1).to(device)
        embeds = self.embedding(word)

        # captions generated 
        captions = []
        
        for i in range(max_len):
            # take encoder output and compute the current attention
            alpha,context = self.attention(features, h)
            
            #store the apla score
            alphas.append(alpha.cpu().detach().numpy())
            
            # conatenation of current state embeddings and context vectors
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size,-1)
            
            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)
            
            #save the generated word
            captions.append(predicted_word_idx.item())
            
            #end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            
            #send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
        
        #covert the vocab idx to words and return sentence
        return [vocab.itos[idx] for idx in captions],alphas
    
    # features from encoder CNN 
    def init_hidden_state(self, encoder_out):
        batch_size = encoder_out.size(0)
        h = torch.zeros(batch_size, self.decoder_dim).to(encoder_out.device)
        c = torch.zeros(batch_size, self.decoder_dim).to(encoder_out.device)
        return h, c
# Seq2Seq model to generate image captions
class EncoderDecoder(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        # encoder doesn't need any params to specify
        self.encoder = EncoderCNN()
        # decoder params need to be specified
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size = len(dataset.vocab),
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )
    
    def forward(self, images, captions):
        # pass the images through encoder to ger feature representations
        features = self.encoder(images)
        # features and captions are passed to decoder
        outputs = self.decoder(features, captions)
        return outputs

model = torch.load(model_path, map_location=device)
my_vocab = torch.load(my_vocab, map_location=device)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Get the file from the request
    file = request.files['file'].read()

    # Preprocess the image
    img = Image.open(io.BytesIO(file))
    img = transforms(img)
    # print(img)
    img = img.unsqueeze(0)
    # print(img)
    features = model.encoder(img.to(device))
    caps,alphas = model.decoder.generate_caption(features,vocab=my_vocab)
    caption = ' '.join(caps)
    caption = caption.replace('<EOS>', '')
    caption = caption.replace('<SOS>', '')
    caption = caption.replace('<UNK>', '')
    # print(caption)
    return render_template('result.html', predicted_text=caption)
    # return caption

if __name__ == '__main__':
    app.run(debug=True)
