"""Reference list (IEEE style). Numbers are assigned in order of first citation.

Only genuine sources: the project's own literature review (docs/stage2_literature_review.md) and
standard method papers. Where the literature review lists a source without authors, no authors are added.
"""
_ALL = {
    "smolen2016": 'J. S. Smolen, D. Aletaha, and I. B. McInnes, "Rheumatoid arthritis," Lancet, vol. 388, no. 10055, pp. 2023–2038, 2016, doi: 10.1016/S0140-6736(16)30173-8.',
    "schiavon2021": 'G. Schiavon, G. Capone, M. Frize, S. Zaffagnini, C. Candrian, and G. Filardo, "Infrared thermography for the evaluation of inflammatory and degenerative joint diseases: a systematic review," Cartilage, vol. 13, no. 2 suppl., pp. 1790S–1801S, 2021, doi: 10.1177/19476035211063862.',
    "ring2012": 'E. F. J. Ring and K. Ammer, "Infrared thermal imaging in medicine," Physiological Measurement, vol. 33, no. 3, pp. R33–R46, 2012, doi: 10.1088/0967-3334/33/3/R33.',
    "zhang2020": 'F. Zhang, V. Bazarevsky, A. Vakunov, A. Tkachenka, G. Sung, C.-L. Chang, and M. Grundmann, "MediaPipe Hands: on-device real-time hand tracking," arXiv:2006.10214, 2020. [Online]. Available: https://arxiv.org/abs/2006.10214',
    "qiao2026": 'Y. Qiao, P. Hamann, C. Robson, and D. Cosker, "Dorsal-to-Joint Temperature Differences (D2JTD) extraction from RGB-T images for remote rheumatoid arthritis monitoring," Biomedical Signal Processing and Control, vol. 112, Art. no. 108482, 2026, doi: 10.1016/j.bspc.2025.108482.',
    "flironepro": 'Teledyne FLIR, "FLIR ONE Pro — product specifications." [Online]. Available: https://www.flir.com/products/flir-one-pro/',
    "sanchez2009": 'F. J. Sanchez-Marin, S. Calixto-Carrera, and C. Villaseñor-Mora, "Novel approach to assess the emissivity of the human skin," Journal of Biomedical Optics, vol. 14, no. 2, Art. no. 024006, 2009, doi: 10.1117/1.3086612.',
    "snekhalatha2015": 'U. Snekhalatha, M. Anburajan, V. Sowmiya, B. Venkatraman, and M. Menaka, "Automated hand thermal image segmentation and feature extraction in the evaluation of rheumatoid arthritis," Proc. Inst. Mech. Eng. H, vol. 229, no. 4, pp. 319–331, 2015, doi: 10.1177/0954411915580809.',
    "pauk2019": 'J. Pauk, A. Wasilewska, and M. Ihnatouski, "Infrared thermography sensor for disease activity detection in rheumatoid arthritis patients," Sensors, vol. 19, no. 16, Art. no. 3444, 2019, doi: 10.3390/s19163444.',
    "jones2018": 'B. Jones, I. Hassan, R. T. Tsuyuki, M. F. Dos Santos, A. S. Russell, and E. Yacyshyn, "Hot joints: myth or reality? A thermographic joint assessment of inflammatory arthritis patients," Clinical Rheumatology, vol. 37, no. 9, pp. 2567–2571, 2018, doi: 10.1007/s10067-018-4108-0.',
    "spalding2008": 'S. J. Spalding, C. K. Kwoh, R. Boudreau et al., "Three-dimensional and thermal surface imaging produces reliable measures of joint shape and temperature: a potential tool for quantifying arthritis," Arthritis Research & Therapy, vol. 10, no. 1, Art. no. R10, 2008, doi: 10.1186/ar2360.',
    "thermojis2022": 'ThermoJIS study (machine-learning thermal joint inflammation score from hand thermography, validated against ultrasound), RMD Open, vol. 8, no. 2, Art. no. e002458, 2022. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC9295660',
    "moreira2017": 'D. G. Moreira, J. T. Costello, C. J. Brito et al., "Thermographic imaging in sports and exercise medicine: a Delphi study and consensus statement on the measurement of human skin temperature," Journal of Thermal Biology, vol. 69, pp. 155–162, 2017, doi: 10.1016/j.jtherbio.2017.07.006.',
    "sensors2026": '"Machine learning calibration of smartphone-based infrared thermal cameras: improved bias and persistent random error," Sensors, vol. 26, no. 4, Art. no. 1295, 2026.',
    "zhang2000": 'Z. Zhang, "A flexible new technique for camera calibration," IEEE Trans. Pattern Anal. Mach. Intell., vol. 22, no. 11, pp. 1330–1334, 2000, doi: 10.1109/34.888718.',
    "lugaresi2019": 'C. Lugaresi et al., "MediaPipe: a framework for building perception pipelines," arXiv:1906.08172, 2019. [Online]. Available: https://arxiv.org/abs/1906.08172',
    "otsu1979": 'N. Otsu, "A threshold selection method from gray-level histograms," IEEE Trans. Syst., Man, Cybern., vol. 9, no. 1, pp. 62–66, 1979, doi: 10.1109/TSMC.1979.4310076.',
    "rother2004": 'C. Rother, V. Kolmogorov, and A. Blake, "\'GrabCut\': interactive foreground extraction using iterated graph cuts," ACM Trans. Graphics, vol. 23, no. 3, pp. 309–314, 2004, doi: 10.1145/1015706.1015720.',
    "bradski2000": 'G. Bradski, "The OpenCV library," Dr. Dobb\'s Journal of Software Tools, 2000.',
    "ledoit2004": 'O. Ledoit and M. Wolf, "A well-conditioned estimator for large-dimensional covariance matrices," Journal of Multivariate Analysis, vol. 88, no. 2, pp. 365–411, 2004, doi: 10.1016/S0047-259X(03)00096-4.',
    "rousseeuw1999": 'P. J. Rousseeuw and K. Van Driessen, "A fast algorithm for the minimum covariance determinant estimator," Technometrics, vol. 41, no. 3, pp. 212–223, 1999, doi: 10.1080/00401706.1999.10485670.',
    "liu2008": 'F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in Proc. IEEE Int. Conf. Data Mining (ICDM), 2008, pp. 413–422, doi: 10.1109/ICDM.2008.17.',
    "scholkopf2001": 'B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C. Williamson, "Estimating the support of a high-dimensional distribution," Neural Computation, vol. 13, no. 7, pp. 1443–1471, 2001, doi: 10.1162/089976601750264965.',
    "pedregosa2011": 'F. Pedregosa et al., "Scikit-learn: machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.',
    "dice1945": 'L. R. Dice, "Measures of the amount of ecologic association between species," Ecology, vol. 26, no. 3, pp. 297–302, 1945, doi: 10.2307/1932409.',
    "shrout1979": 'P. E. Shrout and J. L. Fleiss, "Intraclass correlations: uses in assessing rater reliability," Psychological Bulletin, vol. 86, no. 2, pp. 420–428, 1979, doi: 10.1037/0033-2909.86.2.420.',
    "koo2016": 'T. K. Koo and M. Y. Li, "A guideline of selecting and reporting intraclass correlation coefficients for reliability research," Journal of Chiropractic Medicine, vol. 15, no. 2, pp. 155–163, 2016, doi: 10.1016/j.jcm.2016.02.012.',
    "weir2005": 'J. P. Weir, "Quantifying test-retest reliability using the intraclass correlation coefficient and the SEM," Journal of Strength and Conditioning Research, vol. 19, no. 1, pp. 231–240, 2005, doi: 10.1519/15184.1.',
    "bland1986": 'J. M. Bland and D. G. Altman, "Statistical methods for assessing agreement between two methods of clinical measurement," Lancet, vol. 327, no. 8476, pp. 307–310, 1986, doi: 10.1016/S0140-6736(86)90837-8.',
    "holm1979": 'S. Holm, "A simple sequentially rejective multiple test procedure," Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65–70, 1979.',
    "dataset": 'FLIROne hand thermal dataset (17 subjects; RGB, thermal and radiometric recordings of static and open–close hands), dataset README and files, local copy "archive/FLIROne".',
}

_order: list[str] = []


def cite(*keys: str) -> str:
    nums = []
    for k in keys:
        if k not in _ALL:
            raise KeyError(k)
        if k not in _order:
            _order.append(k)
        nums.append(_order.index(k) + 1)
    return "[" + ", ".join(map(str, nums)) + "]"


class _Refs:
    def __iter__(self):
        return iter([dict(key=k, text=_ALL[k]) for k in _order])


REFS = _Refs()


def all_refs():
    """Every reference, cited ones first in citation order (for the supporting list and the slides)."""
    keys = _order + [k for k in _ALL if k not in _order]
    return [dict(key=k, text=_ALL[k]) for k in keys]
