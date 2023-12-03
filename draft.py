# idx_l2 = torch.where(bag_labels == 1)[0]
# if idx_l2.shape[0]>0:
#     att = [data_inst[item][1] for item in idx_l2]
#     neg_bag = []
#     for idx, item in enumerate(att):
#         if 0 < torch.sum(item >= 0.15) < len(item[0]):
#             neg_bag.append(bag[idx_l2[idx]].to(self.device)[(item < 0.15).squeeze()])
#     if len(neg_bag)>0:
#         tmp = torch.zeros(len(bag)+len(neg_bag))
#         tmp[:len(bag)] = bag_labels
#         tmp[len(bag):] = -1
#         tmp.to(self.device)
#         bag_labels = tmp
#         bag = bag + neg_bag
#         data_inst = [self.Attforward(item) for item in bag]
#         idx_l1 = torch.where(bag_labels != 0)[0]
