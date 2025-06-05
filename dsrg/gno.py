import numpy as np

def denormal_order_ints(hbar, ref):

    e_1 = (
            np.einsum("uv,vu->", hbar['a'], ref.gam1['a'], optimize=True)
          + np.einsum("uv,vu->", hbar['b'], ref.gam1['b'], optimize=True)
    )
    print(f"    <HBar_1> = {-e_1}")

    # e_2 = (
    #           0.25 * np.einsum("uvxy,xyuv->", hbar['aa'], ref.rdms['aa'], optimize=True)
    #         + np.einsum("uvxy,xyuv->", hbar['ab'], ref.rdms['ab'], optimize=True)
    #         + 0.25 * np.einsum("uvxy,xyuv->", hbar['bb'], ref.rdms['bb'], optimize=True)
    #         - np.einsum("uvxy,xu,yv->", hbar['aa'], ref.gam1['a'], ref.gam1['a'], optimize=True)
    #         - 2.0 * np.einsum("uvxy,xu,yv->", hbar['ab'], ref.gam1['a'], ref.gam1['b'], optimize=True)
    #         - np.einsum("uvxy,xu,yv->", hbar['bb'], ref.gam1['b'], ref.gam1['b'], optimize=True)
    # )
    e_2 = (
              0.25 * np.einsum("uvxy,xyuv->", hbar['aa'], ref.lambdas['aa'], optimize=True)
            + np.einsum("uvxy,xyuv->", hbar['ab'], ref.lambdas['ab'], optimize=True)
            + 0.25 * np.einsum("uvxy,xyuv->", hbar['bb'], ref.lambdas['bb'], optimize=True)
            - 0.5 * np.einsum("uvxy,xu,yv->", hbar['aa'], ref.gam1['a'], ref.gam1['a'], optimize=True)
            - np.einsum("uvxy,xu,yv->", hbar['ab'], ref.gam1['a'], ref.gam1['b'], optimize=True)
            - 0.5 * np.einsum("uvxy,xu,yv->", hbar['bb'], ref.gam1['b'], ref.gam1['b'], optimize=True)
    )
    print(f"    <HBar_2> = {-e_2}")

    print(f"    Frozen core energy = {ref.e_frozen_core}")
    e0 = -e_1 - e_2

    # hbar['a'] -= (
    #                 np.einsum("vxuy,xy->vu", hbar['aa'], ref.gam1['a'])
    #                 + np.einsum("vxuy,xy->vu", hbar['ab'], ref.gam1['b'])
    # )
    #
    # hbar['b'] -= (
    #                 np.einsum("vxuy,xy->vu", hbar['bb'], ref.gam1['b'])
    #                 + np.einsum("xvyu,xy->vu", hbar['ab'], ref.gam1['a'])
    # )
    hbar['a'] -= (
                    np.einsum("vyux,xy->vu", hbar['aa'], ref.gam1['a'])
                    + np.einsum("vyux,xy->vu", hbar['ab'], ref.gam1['b'])
    )

    hbar['b'] -= (
                    np.einsum("vyux,xy->vu", hbar['bb'], ref.gam1['b'])
                    + np.einsum("yvxu,xy->vu", hbar['ab'], ref.gam1['a'])
    )

    return hbar, e0