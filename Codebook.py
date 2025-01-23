import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(nn.Module):
    def __init__(self, in_chan, num_embeddings, input_chan, embedding_dim, commitment_cost):
        super(Codebook, self).__init__()

        self.in_chan = in_chan
        self._vq = VectorQuantizer(num_embeddings, input_chan **2, embedding_dim, commitment_cost)

    def forward(self, x):
        loss, x_recon, usage, codebook = self._vq(x)
        return loss, x_recon, usage, codebook


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, input_dim, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.input_dim = input_dim
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self.linear1 = nn.Linear(self.input_dim, self._embedding_dim)
        self.linear2 = nn.Linear(self._embedding_dim, self.input_dim)
        self.bn1 = nn.BatchNorm1d(self.input_dim)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        self.bn2 = nn.BatchNorm1d(self._embedding_dim)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
        self.at = nn.SELU()
        self.dropout = nn.Dropout(0.1)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.reshape(-1, self.input_dim)
        flat_input = self.dropout(self.bn1(flat_input))
        flat_input = self.linear1(flat_input)

        # Calculate the distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight)
        quantized = self.dropout(self.bn2(quantized))
        quantized = self.linear2(quantized).reshape(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        usage = torch.mean(encodings, dim=0).tolist()
        codebook = self._embedding.weight.data
        codebook = codebook.cpu().numpy().tolist()
        return loss, quantized, usage, codebook
