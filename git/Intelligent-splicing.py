from PIL import Image
path="./spice/modelX.jpg"
imag=Image.open(path) #读取接头模型图片
img = imag.convert("RGBA")
a=[1,2,3,5,7,9,1,2,3,4] #优化结果输入
path1="./spice/A"+str(a[0])+"UP.JPG"
# path1=str(path1)
print(path1)
# icon=Image.open("./spice/A1UP.JPG")
icon=Image.open(path1)
icon=icon.resize((380,188))
imag.paste(icon,(455,206),mask=None)

icon=Image.open("./spice/00.png")
icon=icon.resize((230,208))
imag.paste(icon,(56,410),mask=None)

icon=Image.open("./spice/9_B_upr.PNG")
icon=icon.resize((380,188))
imag.paste(icon,(866,206),mask=None)

icon=Image.open("./spice/4_C_upr.PNG")
icon=icon.resize((380,188))
imag.paste(icon,(1340,206),mask=None)

icon=Image.open("./spice/1_D_upr.JPG")
icon=icon.resize((380,188))
imag.paste(icon,(1732,206),mask=None)

icon=Image.open("./spice/2_C_mid.JPG")
icon=icon.resize((380,188))
imag.paste(icon,(1380,445),mask=None)


icon=Image.open("./spice/1_D_mid.JPG")
icon=icon.resize((380,188))
imag.paste(icon,(1805,445),mask=None)

icon=Image.open("./spice/1_A_lwr.JPG")
icon=icon.resize((380,188))
imag.paste(icon,(236,740),mask=None)

icon=Image.open("./spice/6_B_lwr.PNG")
icon=icon.resize((380,188))
imag.paste(icon,(802,740),mask=None)

icon=Image.open("./spice/1_C_lwr.JPG")
icon=icon.resize((380,188))
imag.paste(icon,(1270,740),mask=None)

icon=Image.open("./spice/1_D_lwr.JPG")
icon=icon.resize((380,188))
imag.paste(icon,(1800,740),mask=None)

imag.show() #生成优化拼接结果