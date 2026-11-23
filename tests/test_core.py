import unittest

import torch

from network.alignment import fit_orientation_priors, refine_rotation, voxel_centers, weighted_procrustes
from network.features import RGBFeatures, berhu
from network.train import DynamicTaskWeights
from network.weak import PerturbedTopK, candidate_pool, weak_similarities


def yaw(angle):
    a = torch.tensor(angle, dtype=torch.double)
    c, s, zero, one = a.cos(), a.sin(), a.new_tensor(0), a.new_tensor(1)
    return torch.stack((c, zero, s, zero, one, zero, -s, zero, c)).reshape(3, 3)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_equation_two_and_degenerate_overlap(self):
        coarse, residual = yaw(0.7), yaw(0.3)
        scale = torch.tensor([[1.2, 0.7, 0.9]], dtype=torch.double)
        grid = voxel_centers(4, dtype=torch.double).reshape(-1, 3)
        observed = ((grid @ coarse.T) * scale) @ residual.T
        coordinates = observed.T.reshape(1, 3, 4, 4, 4)
        occupied = torch.ones(1, 1, 4, 4, 4, dtype=torch.double)
        result = refine_rotation(coarse[None], scale, occupied, coordinates, occupied)
        torch.testing.assert_close(result[0], residual @ coarse)
        fallback = refine_rotation(coarse[None], scale, occupied, coordinates, occupied * 0)
        torch.testing.assert_close(fallback[0], coarse)

    def test_procrustes_gradient_and_proper_rotation(self):
        points = torch.cat((torch.eye(3), -torch.eye(3))).double()[None]
        observed = (points @ yaw(0.3).T).requires_grad_()
        weights = torch.ones(1, 6, dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(
            lambda target: weighted_procrustes(points, target, weights), (observed,)
        ))
        reflection = observed.detach().clone()
        reflection[..., 0] *= -1
        # Unequal weights make the best proper rotation unique.
        weights[:, 0] = 2
        weights[:, 1] = 3
        proper = weighted_procrustes(points, reflection, weights)
        torch.testing.assert_close(torch.linalg.det(proper), torch.ones(1, dtype=torch.double))

    def test_orientation_medoids_are_category_examples(self):
        rotations = torch.stack([yaw(i / 6) for i in range(20)])
        categories = torch.arange(20) // 10
        priors = fit_orientation_priors(rotations, categories, 2)
        for category in range(2):
            matches = (priors[category, :, None] == rotations[categories == category]).all((-2, -1))
            self.assertTrue(matches.any(1).all())

    def test_weak_labels_choose_views_independently(self):
        obj = torch.tensor([[1.0, 0.0]])
        views = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.6, 0.8], [0.8, 0.6]]])
        cad_geo = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        labels = weak_similarities(obj, views, obj, views.flip(1), obj, cad_geo)
        torch.testing.assert_close(labels["appearance_semantic"], torch.tensor([[1.0, 0.0]]))
        torch.testing.assert_close(labels["combined"], torch.tensor([[0.0, 1.0]]))

    def test_topk_sampling_and_backward(self):
        targets = torch.arange(18.0).reshape(2, 9)
        pool = candidate_pool(targets, 2, 2, 3)
        self.assertEqual(pool.shape, (2, 7))
        self.assertTrue(((pool[:, -4:].sort().values) == torch.arange(4)).all())
        torch.manual_seed(11)
        scores = torch.tensor([[0.05, 0.12, 0.3, 0.08]], requires_grad=True)
        upstream = torch.arange(8.0).reshape(1, 2, 4)
        output = PerturbedTopK.apply(scores, 2, 20, 0.2)
        (output * upstream).sum().backward()
        torch.testing.assert_close(output.sum(-1), torch.ones(1, 2))
        torch.manual_seed(11)
        noise = torch.empty(1, 20, 4).normal_()
        order = (scores.detach()[:, None] + 0.2 * noise).topk(2, dim=-1).indices
        indicators = torch.nn.functional.one_hot(order, num_classes=4).float()
        reference = torch.einsum("bnkd,bne,bkd->be", indicators, noise, upstream) / 4
        torch.testing.assert_close(scores.grad, reference)

    def test_rgb_branch_and_dynamic_loss(self):
        frontend = RGBFeatures().eval()
        rgb = torch.rand(2, 3, 64, 64)
        boxes = torch.tensor([[8, 8, 48, 48], [16, 18, 56, 52]]).float()
        full_mask = torch.ones(2, 64, 64)
        K = torch.tensor([[40.0, 0, 32], [0, 40.0, 32], [0, 0, 1]]).repeat(2, 1, 1)
        measured = torch.stack((torch.ones(1, 64, 64), torch.full((1, 64, 64), 2.0)))
        with torch.no_grad():
            roi, mask, xyz, depth = frontend(rgb, boxes, full_mask, K, torch.tensor([1, 0]), measured)
        self.assertEqual(roi.shape, (2, 256, 16, 16))
        self.assertEqual(mask.shape, (2, 1, 16, 16))
        torch.testing.assert_close(xyz[:, 2].mean((1, 2)), torch.tensor([2.0, 1.0]))
        torch.testing.assert_close(berhu(depth, depth), torch.tensor(0.0))
        dwa = DynamicTaskWeights(start=3)
        self.assertEqual(dwa(torch.tensor([1.0, 2.0]).unbind(), (1, 3), 1).item(), 7)
        dwa(torch.tensor([2.0, 2.0]).unbind(), (1, 3), 2)
        loss = dwa(torch.tensor([3.0, 1.0]).unbind(), (1, 3), 3)
        self.assertGreater(loss.item(), 4)


if __name__ == "__main__":
    unittest.main()
