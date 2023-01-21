# ใช้ GPU สำหรับเทรนโมเดล

นอกจากการใช้ CPU ในการเทรนโมเดลที่เราได้พูดถึงไปแล้ว เรายังสามารถใช้ GPU ในการเทรนโมเดลได้อีกด้วย
โดยการเทรนโมเดลด้วย GPU จะทำให้เราสามารถเทรนโมเดลได้เร็วขึ้นอย่างมาก สำหรับ Pytorch นั้นเราเทรนโมเดลด้วย GPU
ได้โดยการเพิ่มคำสั่ง `.to(device)` ในส่วนของการสร้างโมเดลและการส่งข้อมูลเข้าไปในโมเดล ดังนี้

```py
device = "cuda" if torch.cuda.is_available() else "cpu"  # ตรวจสอบว่ามี GPU พร้อมกับ CUDA หรือไม่
print(device)

model.to(device)  # ส่งโมเดลเข้าไปใน device ที่กำหนด ในที่นี้คือ GPU ถ้ามี
```

จากนั้นระหว่างเทรนโมเดล เราจะทำการส่งข้อมูลเข้าไปใน GPU เช่นกันด้วยคำสั่ง `.to(device)` ดังนี้

```py
for images, labels in train_loader:
    # Change device to cuda!
    # ส่งข้อมูลเข้าไปใน device ที่กำหนด ในที่นี้คือ GPU ถ้ามี
    images, labels = images.to(device), labels.to(device)
    pred = model(images)
    loss = loss_function(pred, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

โดยสิ่งที่ต้องระหว่างเพียงอย่างเดียวคือ**โมเดลและข้อมูลจะต้องอยู่ใน device เดียวกัน**สำหรับเทรนโมเดล (CPU หรือ GPU)
ถ้าเทรนโมเดลแล้วพบ error ว่า

```py
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat1 in method wrapper_addmm)
```

แสดงว่าโมเดลและข้อมูลอาจะไม่อยู่ใน device เดียวกัน ทำให้เกิด error ขึ้นได้ โดยในตัวอย่าง `05_handwritten_gpu.ipynb` ได้ยกตัวอย่างการทำนาย
ที่โมเดลและข้อมูลอยู่คนละ device ทำให้เกิด error นี้ขึ้น

สุดท้ายนั้น การทำนายผล (inference) อาจทำได้ใน GPU แต่บางครั้งเราต้องการส่งผลลัพธ์กลับมาใช้งานใน CPU เราจะใช้คำสั่ง `.cpu()` เพื่อนำผลลัพธ์กลับมาใช้งานใน CPU ได้ ดังนี้

```py
pred = model(img.to(device)) # สมมติโมเดลอยู่ใน GPU เราจะทำการส่งข้อมูลไปใน GPU เพื่อทำนาย
int(pred.detach().cpu().argmax(dim=1)) # จากนั้นทำการ detach จาก GPU และส่งผลลัพธ์กลับมาใช้งานใน CPU
```
