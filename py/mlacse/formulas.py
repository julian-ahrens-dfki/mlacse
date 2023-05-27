import functools

import numpy as np


def euclidean(std, sample):
    return np.sqrt(np.sum((sample - std) ** 2, axis=-1))


def deltaE2000(Labstd, Labsample, kl=1, kc=1, kh=1):
    #function de00 = deltaE2000(Labstd,Labsample, KLCH )
    # Compute the CIEDE2000 color-difference between the sample between a
    # reference
    # with CIELab coordinates Labsample and a standard with CIELab coordinates
    # Labstd
    # The function works on multiple standard and sample vectors too
    # provided Labstd and Labsample are K x 3 matrices with samples and
    # standard specification in corresponding rows of Labstd and Labsample
    # The optional argument KLCH is a 1x3 vector containing the
    # the value of the parametric weighting factors kL, kC, and kH
    # these default to 1 if KLCH is not specified.

    # Based on the article:
    # "The CIEDE2000 Color-Difference Formula: Implementation Notes,
    # Supplementary Test Data, and Mathematical Observations,", G. Sharma,
    # W. Wu, E. N. Dalal, Color Research and Application, vol. 30. No. 1, pp.
    # 21-30, February 2005.
    # available at http://www.ece.rochester.edu/~/gsharma/ciede2000/

    #de00 = [];

    # Error checking to ensure that sample and Std vectors are of correct sizes
    #v=size(Labstd); w = size(Labsample);
    #if ( v(1) ~= w(1) | v(2) ~= w(2) )
    #  disp('deltaE00: Standard and Sample sizes do not match');
    #  return
    #end % if ( v(1) ~= w(1) | v(2) ~= w(2) )
    #if ( v(2) ~= 3)
    #  disp('deltaE00: Standard and Sample Lab vectors should be Kx3 vectors');
    #  return
    #end

    # Parametric factors
    #if (nargin <3 )
    #     % Values of Parametric factors not specified use defaults
    #     kl = 1; kc=1; kh =1;
    #else
    #     % Use specified Values of Parametric factors
    #     if ( (size(KLCH,1) ~=1) | (size(KLCH,2) ~=3))
    #       disp('deltaE00: KLCH must be a 1x3 vector');
    #       return;
    #    else
    #       kl =KLCH(1); kc=KLCH(2); kh =KLCH(3);
    #     end
    #end

    Lstd = Labstd[..., 0]
    astd = Labstd[..., 1]
    bstd = Labstd[..., 2]
    Cabstd = np.sqrt(astd ** 2 + bstd ** 2)

    Lsample = Labsample[..., 0]
    asample = Labsample[..., 1]
    bsample = Labsample[..., 2]
    Cabsample = np.sqrt(asample ** 2 + bsample ** 2)

    Cabarithmean = (Cabstd + Cabsample) / 2

    G = 0.5 * (
            1 - np.sqrt((Cabarithmean ** 7) / (Cabarithmean ** 7 + 25 ** 7)))

    apstd = (1 + G) * astd  # aprime in paper
    apsample = (1 + G) * asample  # aprime in paper
    Cpsample = np.sqrt(apsample ** 2 + bsample ** 2)
    Cpstd = np.sqrt(apstd ** 2 + bstd ** 2)
    # Compute product of chromas and locations at which it is zero for use
    # later
    Cpprod = (Cpsample * Cpstd)
    zcidx = (Cpprod == 0)

    # Ensure hue is between 0 and 2pi
    # NOTE: MATLAB already defines atan2(0,0) as zero but explicitly set it
    # just in case future definitions change
    hpstd = np.arctan2(bstd, apstd)
    hpstd = hpstd + 2 * np.pi * (hpstd < 0)  # rollover ones that come -ve
    hpstd[((abs(apstd) + abs(bstd)) == 0)] = 0
    hpsample = np.arctan2(bsample, apsample)
    hpsample = hpsample + 2 * np.pi * (hpsample < 0)
    hpsample[((abs(apsample) + abs(bsample)) == 0)] = 0

    dL = (Lsample - Lstd)
    dC = (Cpsample - Cpstd)
    # Computation of hue difference
    dhp = (hpsample - hpstd)
    dhp = dhp - 2 * np.pi * (dhp > np.pi)
    dhp = dhp + 2 * np.pi * (dhp < (-np.pi))
    # set chroma difference to zero if the product of chromas is zero
    dhp[zcidx] = 0

    # Note that the defining equations actually need
    # signed Hue and chroma differences which is different
    # from prior color difference formulae

    dH = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
    #dH2 = 4 * Cpprod * (np.sin(dhp / 2)) ** 2

    # weighting functions
    Lp = (Lsample + Lstd) / 2
    Cp = (Cpstd + Cpsample) / 2
    # Average Hue Computation
    # This is equivalent to that in the paper but simpler programmatically.
    # Note average hue is computed in radians and converted to degrees only
    # where needed
    hp = (hpstd + hpsample) / 2
    # Identify positions for which abs hue diff exceeds 180 degrees
    hp = hp - (abs(hpstd - hpsample) > np.pi) * np.pi
    # rollover ones that come -ve
    hp = hp + (hp < 0) * 2 * np.pi
    # Check if one of the chroma values is zero, in which case set
    # mean hue to the sum which is equivalent to other value
    hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]

    Lpm502 = (Lp - 50) ** 2
    Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
    Sc = 1 + 0.045 * Cp
    T = (1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp)
            + 0.32 * np.cos(3 * hp + np.pi / 30)
            - 0.20 * np.cos(4 * hp - 63 * np.pi / 180))
    Sh = 1 + 0.015 * Cp * T
    delthetarad = (
            30 * np.pi / 180) * np.exp(-((180 / np.pi * hp - 275) / 25) ** 2)
    Rc = 2 * np.sqrt((Cp ** 7) / (Cp ** 7 + 25 ** 7))
    RT = -np.sin(2 * delthetarad) * Rc

    klSl = kl * Sl
    kcSc = kc * Sc
    khSh = kh * Sh

    # The CIE 00 color difference
    de00 = np.sqrt((dL / klSl) ** 2 + (dC / kcSc) ** 2 + (dH / khSh) ** 2
            + RT * (dC / kcSc) * (dH / khSh))

    return de00


def cmc(Labstd, Labsample, l=1, c=1, symmetric=False):

    Lstd = Labstd[..., 0]
    astd = Labstd[..., 1]
    bstd = Labstd[..., 2]
    Cabstd = np.sqrt(astd ** 2 + bstd ** 2)

    Lsample = Labsample[..., 0]
    asample = Labsample[..., 1]
    bsample = Labsample[..., 2]
    Cabsample = np.sqrt(asample ** 2 + bsample ** 2)

    # Compute product of chromas and locations at which it is zero for use
    # later
    Cabprod = (Cabsample * Cabstd)
    zcidx = (Cabprod == 0)

    # Ensure hue is between 0 and 2pi
    # NOTE: MATLAB already defines atan2(0,0) as zero but explicitly set it
    # just in case future definitions change
    hstd = np.arctan2(bstd, astd)
    hstd = hstd + 2 * np.pi * (hstd < 0)  # rollover ones that come -ve
    hstd[((abs(astd) + abs(bstd)) == 0)] = 0
    hsample = np.arctan2(bsample, asample)
    hsample = hsample + 2 * np.pi * (hsample < 0)
    hsample[((abs(asample) + abs(bsample)) == 0)] = 0

    dL = (Lsample - Lstd)
    dC = (Cabsample - Cabstd)
    # Computation of hue difference
    dh = (hsample - hstd)
    dh = dh - 2 * np.pi * (dh > np.pi)
    dh = dh + 2 * np.pi * (dh < (-np.pi))
    # set chroma difference to zero if the product of chromas is zero
    dh[zcidx] = 0

    dH = 2 * np.sqrt(Cabprod) * np.sin(dh / 2)
    #dH2 = 4 * Cabprod * (np.sin(dh / 2)) ** 2

    # weighting functions
    if symmetric:
        Lp = (Lsample + Lstd) / 2
        Cp = (Cabstd + Cabsample) / 2
        # Average Hue Computation
        # This is equivalent to that in the paper but simpler programmatically.
        # Note average hue is computed in radians and converted to degrees only
        # where needed
        hp = (hstd + hsample) / 2
        # Identify positions for which abs hue diff exceeds 180 degrees
        hp = hp - (abs(hstd - hsample) > np.pi) * np.pi
        # rollover ones that come -ve
        hp = hp + (hp < 0) * 2 * np.pi
        # Check if one of the chroma values is zero, in which case set
        # mean hue to the sum which is equivalent to other value
        #hp[zcidx] = hsample[zcidx] + hstd[zcidx]
        hp[zcidx] = (hsample + hstd)[zcidx]
    else:
        Lp = Lstd
        Cp = Cabstd
        hp = hstd

    Sl = np.where(Lp < 16, 0.511, 0.040975 * Lp / (1 + 0.01765 * Lp))
    Sc = 0.0638 * Cp / (1 + 0.0131 * Cp) + 0.638
    F = np.sqrt(Cp ** 4 / (Cp ** 4 + 1900))
    T = np.where((164 * np.pi / 180 <= hp) & (hp <= 345 * np.pi / 180),
            0.56 + abs(0.2 * np.cos(hp + 168 * np.pi / 180)),
            0.36 + abs(0.4 * np.cos(hp + 35 * np.pi / 180)))
    Sh = Sc * (F * T + 1 - F)

    lSl = l * Sl
    cSc = c * Sc

    # The CMC color difference
    decmc = np.sqrt((dL / lSl) ** 2 + (dC / cSc) ** 2 + (dH / Sh) ** 2)

    return decmc


cmc_symmetric = functools.partial(cmc, symmetric=True)


def cie94(Labstd, Labsample, K1=0.045, K2=0.015, kl=1, kc=1, kh=1,
        symmetric=False):

    Lstd = Labstd[..., 0]
    astd = Labstd[..., 1]
    bstd = Labstd[..., 2]
    Cabstd = np.sqrt(astd ** 2 + bstd ** 2)

    Lsample = Labsample[..., 0]
    asample = Labsample[..., 1]
    bsample = Labsample[..., 2]
    Cabsample = np.sqrt(asample ** 2 + bsample ** 2)

    # Compute product of chromas and locations at which it is zero for use
    # later
    Cabprod = (Cabsample * Cabstd)
    zcidx = (Cabprod == 0)

    # Ensure hue is between 0 and 2pi
    # NOTE: MATLAB already defines atan2(0,0) as zero but explicitly set it
    # just in case future definitions change
    hstd = np.arctan2(bstd, astd)
    hstd = hstd + 2 * np.pi * (hstd < 0)  # rollover ones that come -ve
    hstd[((abs(astd) + abs(bstd)) == 0)] = 0
    hsample = np.arctan2(bsample, asample)
    hsample = hsample + 2 * np.pi * (hsample < 0)
    hsample[((abs(asample) + abs(bsample)) == 0)] = 0

    dL = (Lsample - Lstd)
    dC = (Cabsample - Cabstd)
    # Computation of hue difference
    dh = (hsample - hstd)
    dh = dh - 2 * np.pi * (dh > np.pi)
    dh = dh + 2 * np.pi * (dh < (-np.pi))
    # set chroma difference to zero if the product of chromas is zero
    dh[zcidx] = 0

    dH = 2 * np.sqrt(Cabprod) * np.sin(dh / 2)
    #dH2 = 4 * Cabprod * (np.sin(dh / 2)) ** 2

    # weighting functions
    if symmetric:
        Cp = (Cabstd + Cabsample) / 2
    else:
        Cp = Cabstd

    Sl = 1
    Sc = 1 + K1 * Cp
    Sh = 1 + K2 * Cp

    klSl = kl * Sl
    kcSc = kc * Sc
    khSh = kh * Sh

    # The CIE 94 color difference
    de94 = np.sqrt((dL / klSl) ** 2 + (dC / kcSc) ** 2 + (dH / khSh) ** 2)

    return de94


cie94_symmetric = functools.partial(cie94, symmetric=True)


