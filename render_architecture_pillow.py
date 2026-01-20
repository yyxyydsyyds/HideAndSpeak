from PIL import Image, ImageDraw, ImageFont

W, H = 1400, 800
img = Image.new('RGB', (W, H), 'white')
d = ImageDraw.Draw(img)
font = ImageFont.load_default()

def draw_box(x, y, w, h, title, fill='#eef', outline='black'):
    d.rectangle([x, y, x+w, y+h], fill=fill, outline=outline)
    # text wrap
    lines = title.split('\n')
    tx = x + 8
    ty = y + 8
    for line in lines:
        d.text((tx, ty), line, fill='black', font=font)
        ty += 14

def arrow(x1,y1,x2,y2):
    d.line((x1,y1,x2,y2), fill='black', width=2)
    # simple triangle head
    import math
    ang = math.atan2(y2-y1, x2-x1)
    sz = 8
    p1 = (x2 - sz*math.cos(ang - 0.3), y2 - sz*math.sin(ang - 0.3))
    p2 = (x2 - sz*math.cos(ang + 0.3), y2 - sz*math.sin(ang + 0.3))
    d.polygon([p1, p2, (x2,y2)], fill='black')

# positions
x0, y0 = 40, 60
# Inputs
draw_box(x0, y0, 220, 60, 'Carrier\n(raw spectrogram)\n[B, C=1, F, T]')
draw_box(x0, y0+100, 220, 60, 'Msg\n(vector or spectrogram)\n[B, L] or [B, C, F, T]')

# Encoder cluster
ex, ey = 320, 20
draw_box(ex, ey, 260, 180, 'Encoder (En_ac)\nInputProj\nN x TransformerLayer\n-> X_ac [B, D, H, W]')
draw_box(ex, ey+200, 260, 80, 'En_wm\n(Watermark Encoder)\n-> [B, D_wm] or [B,C,h,w]')

# Generator cluster
gx, gy = 640, 60
draw_box(gx, gy, 260, 160, 'GeneratorG\n(MultiScaleTransformerGenerator)\nFusion: Cross-attn/Concat\n-> S [B,C,F,T]')

# Container + Decoder
cx, cy = 960, 40
draw_box(cx, cy, 300, 100, 'Container\ncarrier + S\n[B,C,F,T]')
draw_box(cx, cy+140, 300, 120, 'Decoder / WatermarkDecoder\n-> msg_reconst [B,L\'] (vector)')

# Losses
lx, ly = 540, 300
draw_box(lx, ly, 320, 120, 'Losses (training):\ncarrier_loss, msg_loss (use wm_ae for transformer),\npert_loss (regularize S)')

# dashed wm_ae box (training-only)
wax, way = 320, 320
draw_box(wax, way, 240, 80, 'wm_ae (training-only)\nEn_wm + MLP -> maps msg -> vector target', fill='#efe')

# arrows
arrow(x0+220, y0+30, ex, ey+30)        # Carrier -> En_ac
arrow(x0+220, y0+130, ex+10, ey+220)   # Msg -> En_wm
arrow(ex+260, ey+60, gx, gy+30)         # X_ac -> Generator
arrow(ex+130, ey+260, gx+20, gy+120)    # En_wm -> Generator
arrow(gx+260, gy+80, cx, cy+40)         # S -> Container
arrow(cx+150, cy+140, cx+150, cy+140+10) # small connector
arrow(cx+150, cy+140+10, cx+150, cy+140+40) # Container -> Decoder
arrow(cx+150, cy+260, lx+160, ly+30)    # msg_reconst -> Losses
arrow(cx+20, cy+50, lx+10, ly+30)       # Container -> Losses
arrow(wax+240, way+40, lx+50, ly+40)    # wm_ae -> Losses

# labels
d.text((20, 6), 'Transformer Watermarking Architecture', fill='black', font=font)

img.save('architecture_transformer_pillow.png')
print('Saved: architecture_transformer_pillow.png')
