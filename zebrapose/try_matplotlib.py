import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
def add_box(ax):
    '''用红框标出一个ax的范围.'''
    axpos = ax.get_position()
    rect = mpl.patches.Rectangle(
        (axpos.x0, axpos.y0), axpos.width, axpos.height,
        lw=3, ls='--', ec='r', fc='none', alpha=0.5,
        transform=ax.figure.transFigure
    )
    ax.patches.append(rect)

def add_right_cax(ax, pad, width):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距.
    width是cax的宽度.
    '''
    axpos = ax.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax

def test_data():
    '''生成测试数据.'''
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-X**2) + np.exp(-Y**2)
    # 将Z缩放至[0, 100].
    Z = (Z - Z.min()) / (Z.max() - Z.min()) * 100

    return X, Y, Z

X, Y, Z = test_data()
cmap = mpl.cm.viridis

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

# 提前用红框圈出每个ax的范围,并关闭刻度显示.
for ax in axes.flat:
    add_box(ax)
    ax.axis('off')

# 第一个子图中不画出colorbar.
im = axes[0, 0].pcolormesh(X, Y, Z, cmap=cmap, shading='nearest')
axes[0, 0].set_title('without colorbar')

# 第二个子图中画出依附于ax的垂直的colorbar.
im = axes[0, 1].pcolormesh(X, Y, Z, cmap=cmap, shading='nearest')
cbar = fig.colorbar(im, ax=axes[0, 1], orientation='vertical')
axes[0, 1].set_title('add vertical colorbar to ax')

# 第三个子图中画出依附于ax的水平的colorbar.
im = axes[1, 0].pcolormesh(X, Y, Z, cmap=cmap, shading='nearest')
cbar = fig.colorbar(im, ax=axes[1, 0], orientation='horizontal')
axes[1, 0].set_title('add horizontal colorbar to ax')

# 第三个子图中将垂直的colorbar画在cax上.
im = axes[1, 1].pcolormesh(X, Y, Z, cmap=cmap, shading='nearest')
cax = add_right_cax(axes[1, 1], pad=0.02, width=0.02)
cbar = fig.colorbar(im, cax=cax)
axes[1, 1].set_title('add vertical colorbar to cax')

plt.show()