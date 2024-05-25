import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel

class MultiHeadedDotAttention(nn.Module):
    def __init__(self, num_heads, features_size, dropout=0.1):
        super(MultiHeadedDotAttention, self).__init__()
        self.d_model = features_size
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads

        # Create linear projections
        self.query_linear = nn.Linear(features_size, features_size)
        self.key_linear = nn.Linear(features_size, features_size)
        self.value_linear = nn.Linear(features_size, features_size)

        self.aoa_layer = nn.Sequential(
            nn.Linear(features_size * 2, features_size * 2),
            nn.GLU()
        )
        self.output_linear = nn.Linear(features_size, features_size)

        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, dropout, att_mask = None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if att_mask is not None:
          scores = scores.masked_fill(att_mask[:, None, None, :] == 1, float('-inf'))
          print(f'Scores : {scores}')
        p_attn = F.softmax(scores, dim=-1)
        p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value)

    def forward(self, query, key, value, att_mask = None):
        batch_size = query.size(0)

        query_ = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key_ = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value_ = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attended = self.attention(query_, key_, value_, self.dropout, att_mask)

        # Concat using view
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, -1, self.d_model)

        # Attention on Attention

        aoa_output = self.aoa_layer(torch.cat([attended, query], dim = 2))

        return self.output_linear(aoa_output)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualConnection(nn.Module):
    def __init__(self, _size, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(_size)

    def forward(self, x, att_features):
        return x + self.dropout(self.norm(att_features))

class AoA_Refiner_Layer(nn.Module):
    def __init__(self, features_size, num_heads, dropout=0.1):
        super(AoA_Refiner_Layer, self).__init__()
        self.attn = MultiHeadedDotAttention(num_heads, features_size)
        self.res_connection = ResidualConnection(features_size, dropout)

    def forward(self, x):
        att_features = self.attn(x, x, x)
        refined_features = self.res_connection(x, att_features)

        return refined_features


class AoA_Refiner_Core(nn.Module):
    def __init__(self, num_heads, stack_layers, features_size, embedding_size):
        super(AoA_Refiner_Core, self).__init__()
        self.layers = nn.ModuleList([AoA_Refiner_Layer(features_size, num_heads) for _ in range(stack_layers)])
        self.layer = AoA_Refiner_Layer(features_size, num_heads)
        self.norm = LayerNorm(features_size)
        self.rezied_features = nn.Linear(features_size, embedding_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # x = self.layer(x)
        return self.rezied_features(self.norm(x))

class AoA_Decoder_Core(nn.Module):
  def __init__(self, embedding_layer, num_heads, embedding_size, vocab_size):
    super(AoA_Decoder_Core, self).__init__()
    self.embedding_layer = embedding_layer
    self.att_lstm = nn.LSTM(embedding_size * 2, embedding_size, num_layers = 2)
    self.multi_head = MultiHeadedDotAttention(num_heads, embedding_size)
    self.residual_connect = ResidualConnection(embedding_size)
    self.out_linear = nn.Linear(embedding_size, vocab_size)

    self.out_dropout = nn.Dropout(0.1)
    self.embedding_size = embedding_size


  def forward(self, features, captions_ids, captions_mask = None):
    batch_size = features.size(0)
    sequence_length = captions_ids.size(1)

    features_ = torch.mean(features, dim = 1).unsqueeze(dim=1).expand(batch_size, sequence_length, self.embedding_size) # 32, 56, 768
    embedded_captions = self.embedding_layer(captions_ids) # 32, 56, 768

    input_concat = torch.cat([features_, embedded_captions], dim = 2)
    output, (h_att, c_att) = self.att_lstm(input_concat) # batch_size, 56, 2048

    att = self.multi_head(output, features, features)

    residual_aoa = self.residual_connect(output, att)

    output = self.out_linear(residual_aoa)

    return self.out_dropout(output)

class AoA_Model(nn.Module):
  def __init__(self, language_name, features_size, device = 'cpu', num_heads = 8, stack_layers = 6, crossentropy_output=True):
    super(AoA_Model, self).__init__()

    self.device = device
    self.crossentropy_output = crossentropy_output

    # Language Components
    language_model = AutoModel.from_pretrained(language_name)

    embedding_layer = language_model.embeddings.to(self.device)
    vocab_size = embedding_layer.word_embeddings.num_embeddings
    embedding_size = embedding_layer.word_embeddings.embedding_dim

    self.refiner_layer = AoA_Refiner_Core(num_heads, stack_layers, features_size, embedding_size)
    self.decoder_layer = AoA_Decoder_Core(embedding_layer, num_heads, embedding_size, vocab_size)

    self.padding_idx = language_model.config.pad_token_id

  def _shift_left(self, input_ids):

      shifted_input_ids = input_ids.roll(shifts=-1, dims=1).to(self.device)
      shifted_input_ids[:, -1] = self.padding_idx

      return shifted_input_ids

  def forward(self, samples, captions_mask = None):
    features = samples['obj_features'].to(self.device)
    captions_ids = samples['labels'].to(self.device)

    refined_features = self.refiner_layer(features)
    decoded_outputs = self.decoder_layer(refined_features, captions_ids, captions_mask)

    if self.crossentropy_output:
      output_ = decoded_outputs.transpose(-1, 1)

    ground_truth = self._shift_left(captions_ids)

    return (output_, ground_truth)
