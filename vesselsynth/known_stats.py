"""
References
----------
..[1] "Branching patterns for arterioles and venules of the human cerebral cortex"
      Francis Cassot, Frederic Lauwers, Sylvie Lorthois,
      Prasanna Puwanarajah, Valérie Cances-Lauwers, Henri Duvernoy
      Brain Research (2010)
..[2] "Tortuosity and other vessel attributes for arterioles and venules
       of the human cerebral cortex"
      Sylvie Lorthois, Frederic Lauwers, Francis Cassot
      Microvascular Research (2014)
..[3] "Morphometry of pig coronary arterial trees"
      Kassab, G. S ; Rider, C. A ; Tang, N. J ; Fung, Y. C
      American journal of physiology (2006)
..[4] "A Novel Three-Dimensional Computer-Assisted Method for a Quantitative
       Study of Microvascular Networks of the Human Cerebral Cortex"
      Francis Cassot, Frederic Lauwers, Celine Fouard,
      Steffen Prohaska, Valérie Lauwers-Cances
      Microcirculation (2006)
"""


# =============================================================================
#                           STATISTICS FROM
# "Branching patterns for arterioles and venules of the human cerebral cortex"
# =============================================================================
# cortical surface: 1.5 cm^2
# tissue volume: 28.6 mm^3

prop_arterioles_venules = [0.577, 0.423]
prop_order_1234 = [0.713, 0.217, 0.666, 0.036]
prop_bifurcation_type = [0.54, 0.374, 0.86]  # ALB / HSB / IEB

# Kassab's arterial tree taxonomy
# -------------------------------
# order 0 = terminal vessel (precapilary)
# order N (large N) = ascending/descending vessel
#
#   when two vessels (A1, A2) or order n meet, their parent (A0) is of
#   order n+1 iff
#        D(A0) > ( mean(D[n]) + sd(D[n]) + mean(D[n+1]) - sd(D[n+1]) ) / 2
#
# ALB = asymmetric-lateral bifurcation
#     = a lower order vessel branches laterally between two segments of the same element
# HSB = homogenous-symmetrical bifurcation
#     = the order of the two daughter branches equals that of the parent vessel minus one
# IEB = inter-element bifurcation
#     = the three branches have the same order
#
#         ALB                      HSB                  IEB
#         ===                       ||                   |
#          |                        /\                  / \

area_ratio_children_parent = {
    'all': dict(mean=1.519, median=1.626, sd=0.684),
    'arterioles': dict(mean=1.6602, median=1.5495, sd=0.7135),
    'venules': dict(mean=1.5787, median=1.4691, sd=0.6394),
    '1': dict(mean=1.6378, median=1.5545, sd=0.6055),
    '2': dict(mean=1.5365, median=1.3815, sd=0.7161),
    '3': dict(mean=1.7943, median=1.5283, sd=1.1693),
    '4': dict(mean=1.4954, median=1.4083, sd=0.5382),
    'alb': {
        'all': dict(mean=1.6346, median=1.5026, sd=0.5382),
        '1': dict(mean=1.5948, median=1.5052, sd=0.6089),
        '2': dict(mean=1.6307, median=1.4576, sd=0.7561),
        '3': dict(mean=1.9089, median=1.621, sd=1.267),
        '4': dict(mean=1.4984, median=1.4083, sd=0.5853),
    },
    'hsb': {
        'all': dict(mean=1.5574, nedian=1.4934, sd=0.5691),
        '1': dict(mean=1.6386, median=1.5786, sd=0.5728),
        '2': dict(mean=1.2189, median=1.1693, sd=0.4194),
        '3': dict(mean=1.3335, median=1.2676, sd=0.4556),
        '4': dict(mean=1.4814, median=1.4194, sd=0.2481),
    },
    'ieb': {
        'all': dict(mean=1.8701, median=1.7116, sd=0.6999),
        '1': dict(mean=1.824, median=1.6641, sd=0.6753),
        '2': dict(mean=2.2389, median=2.1514, sd=0.7694),
        '3': dict(mean=2.9288, median=2.8946, sd=0.8417),
        '4': None,
    },
}
area_ratio_child1_parent = {
    'all': dict(mean=0.905, median=0.984, sd=0.46),
    'arterioles': dict(mean=1.0039, median=0.9198, sd=0.4772),
    'venules': dict(mean=0.9560, median=0.8864, sd=0.43339),
    '1': dict(mean=0.9579, nedian=0.8953, sd=0.3950),
    '2': dict(mean=0.9984, median=0.8952, sd=0.5026),
    '3': dict(mean=1.2049, median=1.0228, sd=0.7871),
    '4': dict(mean=1.0794, median=1.0000, sd=0.3884),
    'alb': {
        'all': dict(mean=1.0247, median=0.9303, sd=0.5073),
        '1': dict(mean=0.9548, median=0.8865, sd=0.4016),
        '2': dict(mean=1.0835, median=0.9634, sd=0.5262),
        '3': dict(mean=1.3068, median=1.1006, sd=0.8436),
        '4': dict(mean=1.1262, median=1.0134, sd=0.4109),
    },
    'hsb': {
        'all': dict(mean=0.8954, median=0.8430, sd=0.3513),
        '1': dict(mean=0.9299, median=0.8821, sd=0.3547),
        '2': dict(mean=0.7491, median=0.6976, sd=0.3039),
        '3': dict(mean=0.8100, median=0.7440, sd=0.2993),
        '4': dict(mean=0.8611, median=0.8660, sd=0.1211),
    },
    'ieb': {
        'all': dict(mean=1.1101, median=0.9890, sd=0.4976),
        '1': dict(mean=1.0801, median=0.9647, sd=0.4824),
        '2': dict(mean=1.3531, median=1.2337, sd=0.5523),
        '3': dict(mean=1.7349, median=1.7278, sd=0.4549),
        '4': None,
    },
}
area_ratio_child2_parent = {
    'all': dict(mean=0.608, median=0.642, sd=0.294),
    'arterioles': dict(mean=0.6564, median=0.6217, sd=0.3042),
    'venules': dict(mean=0.6227, median=0.5906, sd=0.2787),
    '1': dict(mean=0.6799, median=0.6526, sd=0.2624),
    '2': dict(mean=0.5382, median=0.4822, sd=0.2923),
    '3': dict(mean=0.5893, median=0.4902, sd=0.4734),
    '4': dict(mean=0.4161, median=0.3815, sd=0.2714),
    'alb': {
        'all': dict(mean=0.6099, median=0.5676, sd=0.3145),
        '1': dict(mean=0.6400, median=0.6102, sd=0.2643),
        '2': dict(mean=0.5472, median=0.4860, sd=0.3147),
        '3': dict(mean=0.6021, median=0.4755, sd=0.5198),
        '4': dict(mean=0.3723, median=0.3447, sd=0.2721),
    },
    'hsb': {
        'all': dict(mean=0.6620, median=0.6364, sd=0.2581),
        '1': dict(mean=0.7087, median=0.6853, sd=0.2534),
        '2': dict(mean=0.4698, median=0.4511, sd=0.1818),
        '3': dict(mean=0.5235, median=0.5006, sd=0.2005),
        '4': dict(mean=0.6203, median=0.6503, sd=0.1593),
    },
    'ieb': {
        'all': dict(mean=0.7600, median=0.7194, sd=0.2694),
        '1': dict(mean=0.7439, median=0.7008, sd=0.2618),
        '2': dict(mean=0.8858, median=0.8954, sd=0.2842),
        '3': dict(mean=1.194, median=1.167, sd=0.3925),
        '4': None,
    },
}
area_ratio_child2_child1 = {
    'mean': dict(mean=0.721, median=0.686, sd=0.213),
    'arterioles': dict(mean=0.6876, median=0.7201, sd=0.2099),
    'venules': dict(mean=0.6841, median=0.7243, sd=0.2176),
    '1': dict(mean=0.7356, median=0.7645, sd=0.1805),
    '2': dict(mean=0.5796, median=0.5742, sd=0.2305),
    '3': dict(mean=0.5187, median=0.5095, sd=0.2482),
    '4': dict(mean=0.4053, median=0.3663, sd=0.2556),
    'alb': {
        'all': dict(mean=0.6303, median=0.6493, sd=0.2266),
        '1': dict(mean=0.6971, median=0.7209, sd=0.1927),
        '2': dict(mean=0.5359, median=0.5091, sd=0.2276),
        '3': dict(mean=0.4769, median=0.4318, sd=0.2466),
        '4': dict(mean=0.3374, median=0.3003, sd=0.2193),
    },
    'hsb': {
        'all': dict(mean=0.7586, median=0.7887, sd=0.1726),
        '1': dict(mean=0.7819, median=0.8096, sd=0.1538),
        '2': dict(mean=0.6673, median=0.6914, sd=0.2125),
        '3': dict(mean=0.6692, median=0.6597, sd=0.1914),
        '4': dict(mean=0.7220, median=0.7758, sd=0.1593),
    },
    'ieb': {
        'all': dict(mean=0.7226, median=0.7448, sd=0.1796),
        '1': dict(mean=0.7258, median=0.7532, sd=0.1799),
        '2': dict(mean=0.6955, median=0.7120, sd=0.1794),
        '3': dict(mean=0.6801, median=0.6784, sd=0.0651),
        '4': None,
    },
}

tree_counts = dict(all=228, arterial=152, venous=76)
segment_counts = dict(all=91056, arterial=11014, venous=8042)
element_counts = dict(all=13596, arterial=7604, venous=5992)
#   "" a segment is a vessel section between two successive
#   "" bifurcations. An element can be made of solitary or various
#   "" vascular segments with the same order (see Section 4.2)
#   "" connected in series
# mean nb nodes per tree: arterioles = 41, venules = 52

bifurcation_counts = {
    'all': 9414,
    # all = arterioles + venules
    'arterioles': 5431, 'venules': 3983,
    # all = alb + hsb + ieb
    'alb': {'all': 5084, '1': 3185, '2': 1380, '3': 491, '4': 28},
    'hsb': {'all': 3524, '1': 2802, '2': 584, '3': 132, '4': 6},
    'ieb': {'all': 806, '1': 732, '2': 79, '3': 4, '4': 0},
    # all = 1 + 2 + 3 + 4
    '1': 6710, '2': 2043, '3': 627, '4': 34,
}
