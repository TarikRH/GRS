# %% [markdown]
# Pipeline de débruitage uniquement

# Imports
import os
import numpy as np
import torch
import dgl
import matplotlib.pyplot as plt
import networkx as nx
import torch.nn.functional as F

from DataHander import DataHandler
from param import args
from models.model import SDNet, GCNModel
from utils import load_model, save_model, fix_random_seed_as
from tqdm import tqdm
from models import diffusion_process as dp
from Utils.Utils import *

import sys
import time
import logging
import pickle

# Fix seed et device
def fix_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dglgraph_to_adjacency_tensor(dgl_graph):
    num_nodes = dgl_graph.num_nodes()
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=dgl_graph.device)
    src, dst = dgl_graph.edges()
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0
    return adj


def adjacency_tensor_to_nx_graph(adj, threshold=0.5):
    adj_np = adj.cpu().numpy()
    G = nx.Graph()
    n = adj_np.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if adj_np[i, j] > threshold:
                G.add_edge(i, j, weight=adj_np[i, j])
    return G

def plot_limited_ego_graph(G, user_id, max_neighbors=10, ax=None, title="Ego-Graph"):
    neighbors = list(G.neighbors(user_id))
    if len(neighbors) > max_neighbors:
        neighbors = np.random.choice(neighbors, max_neighbors, replace=False)
        neighbors = list(neighbors)
    nodes_to_draw = [user_id] + neighbors
    subgraph = G.subgraph(nodes_to_draw)
    if ax is None:
        fig, ax = plt.subplots()
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw(subgraph, pos, ax=ax, node_color='skyblue', edge_color='gray', with_labels=True, node_size=500)
    ax.set_title(title)

def supervised_contrastive_loss(embeddings, adj_matrix, temperature=0.1, alpha=10):
    B = embeddings.size(0)
    emb_norm = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(emb_norm, emb_norm.t())
    mask = torch.eye(B, dtype=torch.bool, device=embeddings.device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    positives = sim_matrix[adj_matrix > 0]
    mask_diag = torch.eye(B, dtype=torch.bool, device=embeddings.device)
    valid_neg_mask = (adj_matrix == 0) & (~mask_diag)
    negatives = sim_matrix[valid_neg_mask]

    pos_loss = -positives.mean() if positives.numel() > 0 else 0.0
    neg_loss = negatives.mean() if negatives.numel() > 0 else 0.0
    loss = alpha * pos_loss + neg_loss

    # For logs
    stats = {
        "n_pos": positives.numel(),
        "n_neg": negatives.numel(),
        "pos_mean": positives.mean().item() if positives.numel() > 0 else float('nan'),
        "neg_mean": negatives.mean().item() if negatives.numel() > 0 else float('nan'),
        "loss": loss.item(),
    }
    return loss, stats

def compute_cosine_similarity_matrix(embeddings):
    norm_emb = F.normalize(embeddings, p=2, dim=1)
    return torch.matmul(norm_emb, norm_emb.T)

# Pipeline principal
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    fix_random_seed_as(args.seed)
    handler = DataHandler()
    handler.LoadData()
    app = Coach(handler)
    app.train()
    # Affichage des graphes finaux (avant/après débruitage)
    user_ids = [0, 1, 2, 3, 4, 5]
    adj_uu_original = dglgraph_to_adjacency_tensor(handler.uu_graph)
    adj_uu_denoised = app.uu_graph_denoised if hasattr(app, 'uu_graph_denoised') else None
    if adj_uu_denoised is None:
        raise ValueError("La matrice uu_graph_denoised n'est pas définie dans le modèle.")
    G_original = adjacency_tensor_to_nx_graph(adj_uu_original, threshold=0.9)
    threshold = 0.5
    adj_uu_denoised_filtered = adj_uu_denoised * adj_uu_original
    G_denoised_filtered = adjacency_tensor_to_nx_graph(adj_uu_denoised_filtered, threshold=0.5)
    for user_id in user_ids:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_limited_ego_graph(G_original, user_id, max_neighbors=10, ax=axes[0], title=f"UU Graph before denoising (user {user_id})")
        if user_id in G_denoised_filtered.nodes:
            plot_limited_ego_graph(G_denoised_filtered, user_id, max_neighbors=10, ax=axes[1], title=f"UU Graph after denoising (user {user_id})")
        else:
            axes[1].set_title(f"UU Graph after denoising (user {user_id})\n(No node {user_id})")
            axes[1].axis('off')
        plt.tight_layout()
        plt.show()
        # Diagnostic textuel
        trust_matrix = handler.trust_matrix if hasattr(handler, 'trust_matrix') else handler.dataset['trust']
        trusted_users_original = set(trust_matrix.getrow(user_id).nonzero()[1])
        if user_id in G_denoised_filtered.nodes:
            retained_users_filtered = set(G_denoised_filtered.neighbors(user_id))
        else:
            retained_users_filtered = set()
        retained_trusted = trusted_users_original & retained_users_filtered
        lost_trusted = trusted_users_original - retained_users_filtered
        new_links = retained_users_filtered - trusted_users_original
        print(f"User {user_id}:")
        print(f"  Original trusted users ({len(trusted_users_original)}): {sorted([int(x) for x in trusted_users_original])}")
        print(f"  Users retained after denoising ({len(retained_users_filtered)}): {sorted([int(x) for x in retained_users_filtered])}")
        print(f"  Trusted users retained ({len(retained_trusted)}): {sorted([int(x) for x in retained_trusted])}")
        print(f"  Trusted users lost ({len(lost_trusted)}): {sorted([int(x) for x in lost_trusted])}")
        print(f"  New links (should be 0) ({len(new_links)}): {sorted([int(x) for x in new_links])}")
        recall = len(retained_trusted) / len(trusted_users_original) if trusted_users_original else 0
        precision = len(retained_trusted) / len(retained_users_filtered) if retained_users_filtered else 0
        print(f"  Recall: {recall:.2%}")
        print(f"  Precision: {precision:.2%}\n")

class Coach:
    def __init__(self, handler):
        self.args = args
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        self.handler = handler
        self.train_loader = self.handler.trainloader
        self.valloader = self.handler.valloader
        self.testloader = self.handler.testloader
        self.n_user, self.n_item = self.handler.n_user, self.handler.n_item
        self.uiGraph = self.handler.ui_graph.to(self.device)
        self.uuGraph = self.handler.uu_graph.to(self.device)
        self.uu_graph_denoised = None  # Stocke le graphe UU débruité
        self.epoch_pos_means = []
        self.epoch_neg_means = []
        self.GCNModel = GCNModel(args, self.n_user, self.n_item).to(self.device)
        output_dims = [args.dims] + [args.n_hid]
        input_dims = output_dims[::-1]
        self.SDNet = SDNet(input_dims, output_dims, args.emb_size, time_type="cat", norm=args.norm).to(self.device)
        self.DiffProcess = dp.DiffusionProcess(args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max, args.steps, self.device).to(self.device)
        self.optimizer1 = torch.optim.Adam([
            {'params': self.GCNModel.parameters(), 'weight_decay': 0},
        ], lr=args.lr)
        self.optimizer2 = torch.optim.Adam([
            {'params': self.SDNet.parameters(), 'weight_decay': 0},
        ], lr=args.difflr)
        self.scheduler1 = torch.optim.lr_scheduler.StepLR(
            self.optimizer1,
            step_size=args.decay_step,
            gamma=args.decay
        )
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(
            self.optimizer2,
            step_size=args.decay_step,
            gamma=args.decay
        )
        self.train_loss = []
        self.his_recall = []
        self.his_ndcg  = []

    def train(self):
        args = self.args
        self.save_history = True
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        log_save = './History/' + args.dataset + '/'
        log_file = args.save_name
        fname = f'{log_file}.txt'
        fh = logging.FileHandler(os.path.join(log_save, fname))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger()
        logger.addHandler(fh)
        logger.info(args)
        logger.info('================')
        best_recall, best_ndcg, best_epoch, wait = 0, 0, 0, 0
        start_time = time.time()
        for self.epoch in range(1, args.n_epoch + 1):
            epoch_losses = self.train_one_epoch()
            self.train_loss.append(epoch_losses)
            print('epoch {} done! elapsed {:.2f}.s, epoch_losses {}'.format(
                self.epoch, time.time() - start_time, epoch_losses
            ), flush=True)
            recall, ndcg = self.test(self.testloader)
            print(f"[Epoch {self.epoch}] Recall@{args.topk}: {recall:.4f}, NDCG@{args.topk}: {ndcg:.4f}")
            similarity_score = self.evaluate_denoising()
            print(f"[Epoch {self.epoch}] Similarity Score = {similarity_score:.4f}")
            self.his_ndcg.append(ndcg)
            cur_best = recall + ndcg > best_recall + best_ndcg
            if cur_best:
                best_recall, best_ndcg, best_epoch = recall, ndcg, self.epoch
                wait = 0
            else:
                wait += 1
            logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
                self.epoch, time.time() - start_time, args.topk, recall, args.topk, ndcg))
            if args.model_dir and cur_best:
                desc = args.save_name
                perf = ''
                fname = f'{args.desc}_{desc}_{perf}.pth'
                save_model(self.GCNModel, self.SDNet, os.path.join(args.model_dir, fname), self.optimizer1, self.optimizer2)
            if self.save_history:
                self.saveHistory()
            if wait >= args.patience:
                print(f'Early stop at epoch {self.epoch}, best epoch {best_epoch}')
                break
        print(f'Best  Recall@{args.topk} {best_recall:.6f}, NDCG@{args.topk} {best_ndcg:.6f}', flush=True)
        print(f"[Epoch {self.epoch}] Similarity Score = {similarity_score:.4f}")
        with torch.no_grad():
            z_u,_ = self.GCNModel(self.uiGraph, self.uuGraph)
            z_u, _ = self.GCNModel(self.uiGraph, self.uuGraph)
            z_u = z_u[:self.n_user]
            t = torch.randint(0, self.args.steps, (self.n_user,), device=self.device)
            noise = torch.randn_like(z_u)
            x_t = self.DiffProcess.q_sample(z_u, t, noise)
            reconstructed_z_u = self.SDNet(x_t, t)
            self.uu_graph_denoised = compute_cosine_similarity_matrix(reconstructed_z_u).cpu()
    def evaluate_denoising(self):
        self.SDNet.eval()
        self.GCNModel.eval()
        with torch.no_grad():
            uiEmbeds, uuEmbeds = self.GCNModel(self.uiGraph, self.uuGraph, True)
            num_samples = 64
            sample_users = torch.randint(0, self.n_user, (num_samples,)).long().to(self.device)
            original_embeddings = uuEmbeds[sample_users]
            reconstructed_embeddings = self.DiffProcess.p_sample(
                self.SDNet, original_embeddings, args.sampling_steps, args.sampling_noise
            )
            cos_sim = F.cosine_similarity(original_embeddings, reconstructed_embeddings, dim=1)
            similarity_score = cos_sim.mean().item()
            print(f"Similarity Score after Denoising: {similarity_score:.4f}")
        return similarity_score
    def train_one_epoch(self):
        self.SDNet.train()
        self.GCNModel.train()
        dataloader = self.train_loader
        epoch_losses = [0] * 3
        dataloader.dataset.negSampling()
        tqdm_dataloader = tqdm(dataloader)
        since = time.time()
        epoch_pos_means = []
        epoch_neg_means = []
        adj_uu = dglgraph_to_adjacency_tensor(self.uuGraph).to(self.device)
        for iteration, batch in enumerate(tqdm_dataloader):
            user_idx, pos_idx, neg_idx = batch
            user_idx = user_idx.long()
            pos_idx = pos_idx.long()
            neg_idx = neg_idx.long()
            user_idx = user_idx.to(self.device)
            pos_idx = pos_idx.to(self.device)
            neg_idx = neg_idx.to(self.device)
            uiEmbeds, uuEmbeds = self.GCNModel(self.uiGraph, self.uuGraph, True)
            uEmbeds = uiEmbeds[:self.n_user]
            iEmbeds = uiEmbeds[self.n_user:]
            user = uEmbeds[user_idx]
            pos = iEmbeds[pos_idx]
            neg = iEmbeds[neg_idx]
            uu_terms = self.DiffProcess.caculate_losses(self.SDNet, uuEmbeds[user_idx], self.args.reweight)
            uuelbo = uu_terms["loss"].mean()
            user = user + uu_terms["pred_xstart"]
            diffloss = uuelbo
            scoreDiff = pairPredict(user, pos, neg)
            bprLoss = - (scoreDiff).sigmoid().log().sum() / self.args.batch_size
            regLoss = ((torch.norm(user) ** 2 + torch.norm(pos) ** 2 + torch.norm(neg) ** 2) * self.args.reg) / self.args.batch_size
            loss = bprLoss + regLoss
            losses = [bprLoss.item(), regLoss.item()]
            loss = diffloss + loss
            losses.append(diffloss.item())
            lambda_contrast = 10
            trust_mask = adj_uu[user_idx][:, user_idx]
            batch_embeddings = uuEmbeds[user_idx]
            loss_contrast, contrast_stats = supervised_contrastive_loss(
                batch_embeddings, trust_mask, temperature=0.1, alpha=10
            )
            loss = loss + lambda_contrast * loss_contrast
            epoch_pos_means.append(contrast_stats['pos_mean'])
            epoch_neg_means.append(contrast_stats['neg_mean'])
            adj_pred = torch.matmul(F.normalize(uuEmbeds, p=2, dim=1), F.normalize(uuEmbeds, p=2, dim=1).T)
            sparsity_lambda = 0.01
            sparsity_loss = torch.abs(adj_pred).sum() / (adj_pred.shape[0] * adj_pred.shape[1])
            loss = loss + sparsity_lambda * sparsity_loss
            beta_recon = 20.0
            batch_uu_embeds = uuEmbeds[user_idx]
            batch_cos_sim = torch.matmul(
                F.normalize(batch_uu_embeds, p=2, dim=1),
                F.normalize(batch_uu_embeds, p=2, dim=1).T
            )
            batch_adj_trust = trust_mask
            recon_loss = F.mse_loss(batch_cos_sim, batch_adj_trust)
            loss = loss + beta_recon * recon_loss
            if iteration == 0:
                n_pos = (trust_mask > 0).sum().item()
                n_neg = (trust_mask == 0).sum().item()
                print(f"[Epoch {self.epoch} | Batch {iteration}] Contrastive loss: {loss_contrast.item():.4f} | Positive pairs: {n_pos} | Negative pairs: {n_neg}")
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()
            epoch_losses = [x + y for x, y in zip(epoch_losses, losses)]
        print(f"[EPOCH SUMMARY] adj_uu sum: {adj_uu.sum().item()}, size: {adj_uu.numel()}, densité: {adj_uu.sum().item() / adj_uu.numel():.6f}")
        if self.scheduler1 is not None:
            self.scheduler1.step()
            self.scheduler2.step()
        epoch_losses = [sum(epoch_losses)] + epoch_losses
        time_elapsed = time.time() - since
        print('Training complete in {:.4f}s'.format(time_elapsed))
        avg_pos = sum(epoch_pos_means) / len(epoch_pos_means)
        avg_neg = sum(epoch_neg_means) / len(epoch_neg_means)
        self.epoch_pos_means.append(avg_pos)
        self.epoch_neg_means.append(avg_neg)
        return epoch_losses
    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg
    def test(self, dataloader):
        self.SDNet.eval()
        self.GCNModel.eval()
        Recall, NDCG = [0] * 2
        num = len(dataloader.dataset)
        since = time.time()
        with torch.no_grad():
            uiEmbeds, uuEmbeds = self.GCNModel(self.uiGraph, self.uuGraph, True)
            tqdm_dataloader = tqdm(dataloader)
            for iteration, batch in enumerate(tqdm_dataloader, start=1):
                user_idx, trnMask = batch
                user_idx = user_idx.long()
                trnMask = trnMask
                uEmbeds = uiEmbeds[:self.n_user]
                iEmbeds = uiEmbeds[self.n_user:]
                user = uEmbeds[user_idx]
                trnMask = trnMask.to(user.device)
                uuemb = uuEmbeds[user_idx]
                user_predict = self.DiffProcess.p_sample(self.SDNet, uuemb, args.sampling_steps, args.sampling_noise)
                user = user + user_predict
                allPreds = torch.mm(user, torch.transpose(iEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
                _, topLocs = torch.topk(allPreds, args.topk)
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), dataloader.dataset.tstLocs, user_idx)
                Recall += recall
                NDCG += ndcg
            time_elapsed = time.time() - since
            print('Testing complete in {:.4f}s'.format(time_elapsed))
            Recall = Recall / num
            NDCG = NDCG / num
        return Recall, NDCG
    def saveHistory(self):
        history = dict()
        history['loss'] = self.train_loss
        history['Recall'] = self.his_recall
        history['NDCG'] = self.his_ndcg
        ModelName = "SDR"
        desc = args.save_name
        perf = ''
        fname = f'{args.desc}_{desc}_{perf}.his'
        with open('./History/' + args.dataset + '/' + fname, 'wb') as fs:
            pickle.dump(history, fs)

if __name__ == "__main__":
    main()

# ...existing code... (Coach, SDNet, GCNModel, etc. doivent rester dans le fichier)

