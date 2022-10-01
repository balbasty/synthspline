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
..[5] "Morphometry of the human cerebral cortex microcirculation:
       general characteristics and space-related profiles"
      Frederic Lauwers, Francis Cassot, Valérie Lauwers-Cances,
      Prasanna Puwanarajah, Henri Duvernoy
      NeuroImage (2008)
..[6] "Robust measures of three-dimensional vascular tortuosity based
       on the minimum curvature of approximating polynomial
       spline fits to the vessel mid-line"
      Michael J. Johnson, Geoff Dougherty
      Med Eng Phys (2007)
..[7] "Methods for Three-Dimensional Geometric Characterization
       of the Arterial Vasculature"
      PADRAIG M. O’FLYNN, GERARD O’SULLIVAN, ABHAY S. PANDIT
      Annals of Biomedical Engineerin (2007)
..[8] "A statistical model of the penetrating arterioles and venules in
       the human cerebral cortex"
      Wahbi K. El-Bouri, Stephen J. Payne
      Microcirculation (2016)
"""


# =============================================================================
#                           STATISTICS FROM
# "Branching patterns for arterioles and venules of the human cerebral cortex"
# =============================================================================


class StatCassot2010:
    # cortical surface: 1.5 cm^2
    # tissue volume: 28.6 mm^3
    cortical_surface = 150    # mm^2
    tissue_volume = 28.6      # mm^3

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

    # area = π r^2 = π d^2 / 4
    #   "" Note that some authors (Kaimovitz et al., 2008) focus on the ratio of
    #   "" diameters. However, the area ratios have a direct significance
    #   "" regarding the dynamics of blood flow because the vessel areas
    #   "" (and not their diameters) are the relevant quantities for mass
    #   "" conservation.
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
            'all': dict(mean=1.5574, median=1.4934, sd=0.5691),
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
        '1': dict(mean=0.9579, median=0.8953, sd=0.3950),
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

    #   "" For each segment i connected to the node, we
    #   "" identified the last vertex inside and the first vertex outside a
    #   "" sphere centered at the node which radius equals the maximal
    #   "" local radius of the three vessels (see Fig. 12) and took the mean
    #   "" of the tangential unit vectors at both points. Let u i be the unit
    #   "" vector obtained this way, where i = 0, 1 or 2 points out,
    #   "" respectively, the mother, major and minor daughter branch
    #   "" of the node. The branching angles were defined as α ij = cos − 1
    #   "" (u i ·u j ). The asymmetry angle was defined as β = α 01 − α 02 .
    #   "" The three branches of a bifurcation define a solid angle Ω
    #   "" equal to the sum of the angles between the three branches:
    #   "" Ω = α 01 + α 02 + α 12 − π. This angle is less than π and equals this
    #   "" value only when the three branches are co-planar. The value
    #   "" of π − Ω, which is called the out-of-plane angle, was used to
    #   "" define the planarity of the bifurcation.

    angle_child1_parent = {
        'all': dict(mean=127.1, median=None, csd=27.3),
        'arterioles': dict(mean=125.81, median=127.89, csd=28.12),
        'venules': dict(mean=128.73, median=131.09, csd=26.01),
        '1': dict(mean=123.42, median=125.7, csd=26.11),
        '2': dict(mean=134.32, median=138.11, csd=27.79),
        '3': dict(mean=141.83, median=146.51, csd=28.04),
        '4': dict(mean=149.00, median=152.595, csd=22.06),
        'alb': {
            'all': dict(mean=129.64, median=132.42, csd=27.92),
            '1': dict(mean=124.73, median=126.99, csd=26.56),
            '2': dict(mean=135.96, median=140.37, csd=27.78),
            '3': dict(mean=143.26, median=148.29, csd=28.79),
            '4': dict(mean=147.99, median=151.66, csd=22.87),
        },
        'hsb': {
            'all': dict(mean=123.93, median=126.48, csd=26.32),
            '1': dict(mean=121.76, median=123.78, csd=25.80),
            '2': dict(mean=131.11, median=134.00, csd=26.94),
            '3': dict(mean=137.21, median=139.72, csd=24.53),
            '4': dict(mean=153.56, median=162.73, csd=17.15),
        },
        'ieb': {
            'all': dict(mean=124.66, median=127.0, csd=25.63),
            '1': dict(mean=124.17, median=126.47, csd=24.91),
            '2': dict(mean=129.41, median=133.02, csd=31.43),
            '3': dict(mean=124.19, median=128.31, csd=20.48),
            '4': None,
        },
    }
    angle_child2_parent = {
        'all': dict(mean=114.5, median=None, csd=26),
        'arterioles': dict(mean=111.64, median=112.19, csd=26.97),
        'venules': dict(mean=118.22, median=120.03, csd=24.06),
        '1': dict(mean=114.45, median=115.65, csd=24.92),
        '2': dict(mean=116.83, median=116.34, csd=10.63),
        '3': dict(mean=113.31, median=113.99, csd=31.34),
        '4': dict(mean=110.92, median=109.92, csd=30.75),
        'alb': {
            'all': dict(mean=113.98, median=114.88, csd=26.58),
            '1': dict(mean=114.17, median=114.97, csd=25.04),
            '2': dict(mean=114.75, median=116.11, csd=27.94),
            '3': dict(mean=110.83, median=110.83, csd=31.56),
            '4': dict(mean=106.72, median=103.94, csd=30.79),
        },
        'hsb': {
            'all': dict(mean=114.70, median=116.17, csd=25.27),
            '1': dict(mean=114.38, median=115.83, csd=24.84),
            '2': dict(mean=114.59, median=116.19, csd=26.27),
            '3': dict(mean=121.47, median=124.59, csd=28.67),
            '4': dict(mean=129.53, median=132.5, csd=22.38),
        },
        'ieb': {
            'all': dict(mean=115.04, median=119.2, csd=17.29),
            '1': dict(mean=115.97, median=118.87, csd=24.60),
            '2': dict(mean=119.36, median=120.54, csd=28.17),
            '3': dict(mean=143.00, median=150.65, csd=27.49),
            '4': None,
        },
    }
    angle_child2_child1 = {
        'mean': dict(mean=103.8, median=None, csd=27.4),
        'arterioles': dict(mean=107.42, median=109.53, csd=28.81),
        'venules': dict(mean=98.97, median=97.95, csd=24.49),
        '1': dict(mean=107.09, median=107.49, csd=26.26),
        '2': dict(mean=97.56, median=96.62, csd=27.83),
        '3': dict(mean=88.90, median=87.53, csd=28.78),
        '4': dict(mean=82.55, median=82.36, csd=23.99),
        'alb': {
            'all': dict(mean=101.86, median=101.88, csd=27.68),
            '1': dict(mean=106.27, median=106.61, csd=26.17),
            '2': dict(mean=96.36, median=95.48, csd=28.11),
            '3': dict(mean=88.84, median=87.81, csd=29.17),
            '4': dict(mean=87.01, median=91.27, csd=23.29),
        },
        'hsb': {
            'all': dict(mean=105.75, median=106.45, csd=27.23),
            '1': dict(mean=107.75, median=108.74, csd=26.79),
            '2': dict(mean=100.06, median=98.85, csd=27.17),
            '3': dict(mean=90.00, median=87.15, csd=26.95),
            '4': dict(mean=62.32, median=65.97, csd=14.76),
        },
        'ieb': {
            'all': dict(mean=107.11, median=106.71, csd=24.97),
            '1': dict(mean=108.15, median=107.5, csd=24.42),
            '2': dict(mean=99.66, median=98.97, csd=26.53),
            '3': dict(mean=59.18, median=65.98, csd=22.13),
            '4': None,
        },
    }

    angle_asymmetry = {
        'mean': dict(mean=12.7, median=None, csd=42.7),
        'arterioles': dict(mean=14.44, median=14.89, csd=44.56),
        'venules': dict(mean=10.43, median=10.11, csd=39.98),
        '1': dict(mean=8.91, median=9.05, csd=41.22),
        '2': dict(mean=20.04, median=21.48, csd=43.82),
        '3': dict(mean=29.93, median=31.99, csd=46.42),
        '4': dict(mean=41.55, median=46.10, csd=45.90),
        'alb': {
            'all': dict(mean=15.93, median=15.72, csd=43.91),
            '1': dict(mean=10.57, median=10.71, csd=41.92),
            '2': dict(mean=21.94, median=23.74, csd=44.80),
            '3': dict(mean=34.37, median=40.19, csd=46.37),
            '4': dict(mean=45.76, median=57.1, csd=47.24),
        },
        'hsb': {
            'all': dict(mean=9.17, median=9.05, csd=40.96),
            '1': dict(mean=7.21, median=6.96, csd=40.62),
            '2': dict(mean=17.08, median=19.62, csd=40.97),
            '3': dict(mean=15.61, median=15.13, csd=43.24),
            '4': dict(mean=24.58, median=33.42, csd=34.44),
        },
        'ieb': {
            'all': dict(mean=8.33, median=8.37, csd=40.57),
            '1': dict(mean=8.29, median=8.35, csd=40.14),
            '2': dict(mean=10.41, median=10.67, csd=44.81),
            '3': dict(mean=-17.81, median=-13.59, csd=15.94),
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


# =============================================================================
#                           STATISTICS FROM
#     "Tortuosity and other vessel attributes for arterioles and venules
#                       of the human cerebral cortex"
# =============================================================================

class StatLorthois2014:
    # same dataset as in Cassot et al. 2006
    # cortical surface: 1.5 cm^2
    # tissue volume: 28.6 mm^3
    cortical_surface = 150    # mm^2
    tissue_volume = 28.6      # mm^3

    # each segment is parameterized by r(s) = [x(s), y(s), z(s)], with s
    # discretized in steps of 0.5 um.
    # First, second and third derivatives are noted r', r", r'"
    # The curvature is defined as: κ = |r"| = sqrt( <r", r"> )
    # The torsion is defined as:   τ = <r', r", r'"> / |r"|^2
    #                                = <r', r", r'"> / <r", r">
    # <a, b> = a . b  is the scalar product of two vectors
    # a x b  is the cross product of two vectors
    # <a, b, c> = <(a x b), c> is the scalar triple product of tree vectors
    #
    # The tortuosity of a segment is quantified by the mean, standard
    # deviation and root mean square of its curvature across vertices.

    length_to_diameter_ratio = {
        'all': dict(mean=10.37, median=7.46, sd=9.41),
        'arterioles': dict(median=8.46),
        'venules': dict(median=6.42),
        '0': dict(median=8.77),
        '1': dict(median=7.),   # approx read from graph
        '2': dict(median=5.2),  # approx read form graph
        '3+': dict(median=4.6),
    }
    # log(LDR)[order] = 0.9397 - 0.0971 * order  (for segments)
    #                   0.9111 + 0.194  * order  (for elements)

    mean_curvature = {
        'all': dict(mean=0.149, median=0.153, sd=0.036),

    }

    sd_curvature = {
        'all': dict(mean=0.093, median=0.094, sd=0.026),
    }

    rms_curvature = {
        'all': dict(mean=0.188, median=0.191, sd=0.052),
    }

    sum_of_angles = {
        'all': dict(mean=0.163, median=0.168, sd=0.045),
    }

    dfm = {
        'all': dict(mean=1.234, median=1.135, sd=0.351),
    }
    # distance metric DM = total path length L of a vessel divided by
    #                      the linear distance between its endpoints Dex
    # distance factor metric = DM - 1


# =============================================================================
#                           STATISTICS FROM
#       "Morphometry of the human cerebral cortex microcirculation:
#           general characteristics and space-related profiles"
# =============================================================================

class StatLauwers2008:
    # same dataset as in Cassot et al. 2006
    # cortical surface: 1.5 cm^2
    # tissue volume: 28.6 mm^3
    cortical_surface = 150    # mm^2
    tissue_volume = 28.6      # mm^3

    # normalizations:
    #   1/sqrt(diameter) is almost normal distributed
    #   ln(length) is almost normal distributed
    # the length of the capillaries segments has roughly the same distribution
    # as the whole network. However, the raw diameter of the capillaries
    # is naturally almost normal.
    segment_diameter = dict(
        raw=dict(mean=7.82, median=7.19, sd=3.52),
        normalized=dict(mean=0.38, median=0.37, sd=0.07),
        capillaries=dict(mean=6.47, median=6.45, sd=1.70),
    )
    segment_length = dict(
        raw=dict(mean=52.67, median=50.38, sd=36.14),
        normalized=dict(mean=3.54, median=3.59, sd=0.96),
        capillaries=dict(mean=52.95, median=36.07, sd=49.75),
    )

