import numpy as np
import h5py as h5

def bl2pol_feed_inds(bls, feeds, pol = np.array(['xx','yy'])):
    '''
        Convert bls in raw_timestream object to pol and feeds in timestream object.

        Parameter:
        ----------
        bls: (N,2) array, baselines
        feeds: (M,) array, feeds
        pol: (2,) array, pols, should only be ['xx', 'yy'] or ['yy', 'xx'], default ['xx','yy']

        Return:
        -------
        pf: (K, 4) list, K is for bls index, 4 is for pol index and feed index range in form of [pol_i, feed_i, pol_j, feed_j]
        bl_select: is used to get rid of the baselines that have used feeds that are not in feeds array. use bls[bl_select] to get good bls

    '''
    bls = np.array(bls)
    feeds = np.array(feeds)
    pol = np.array(pol)
    
    lbls = len(bls)
    lfeeds = len(feeds)

    incongruity = False

    xpol_ind = np.where(pol == 'xx')[0][0]

    pf = []
    bl_select = []
    for bli, blj in bls:
        fi = int(np.ceil(bli/2.))
        fj = int(np.ceil(blj/2.))
        if not (fi in feeds and fj in feeds):
            if not incongruity:
                print('Detected incongruity between feeds and baselines, should not use all of the present baselines!')
                incongruity = True
            bl_select += [False]
            continue
        else:
            bl_select += [True]
        fi_ind = np.where(feeds == fi)[0][0]
        fj_ind = np.where(feeds == fj)[0][0]
        ip = bli%2-1-xpol_ind # bli is odd -> xpol -> ip = xpol_in ~ -xpol_ind ~ 1-1-xpol_ind
        jp = blj%2-1-xpol_ind # blj is even -> ypol -> jp = xpol_in+1 ~ -xpol_ind-1
        pf += [[ip,fi_ind,jp,fj_ind]]
    bl_select = np.array(bl_select)
    return pf, bl_select
if __name__ == '__main__':
    with h5.File('./data/3srcNP_20180101214415_20180101224415.hdf5','r') as filein:
        bls = filein['blorder'][:]
    feed1 = np.arange(16) + 1
    feed2 = np.delete(feed1,[6,13])
    pf1,bls1 = bl2pol_feed_inds(bls, feed1)
    pf2,bls2 = bl2pol_feed_inds(bls, feed2)
    for pf,bl in zip(pf2,bls[bls2]):
        print(pf,bl)
