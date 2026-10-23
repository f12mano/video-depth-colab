# Genjutsu Studio: بديل Higgsfield المفتوح بضغطة زرار واحدة

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/f12mano/video-depth-colab/blob/main/video_depth_colab.ipynb)
[![Higgsfield V2V](https://img.shields.io/badge/Higgsfield_V2V-Reverse_Engineered-black?style=flat&color=32D74B)](https://github.com/f12mano/video-depth-colab/tree/main)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/f12mano/video-depth-colab/tree/main)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  <img src="https://raw.githubusercontent.com/f12mano/Genjustsu-Open-Source-Workflow/main/examples/previews/workflow.gif" width="30%" alt="Workflow" />
  <img src="https://raw.githubusercontent.com/f12mano/Genjustsu-Open-Source-Workflow/main/examples/previews/example-1.gif" width="33%" alt="Example 1" />
  <img src="https://raw.githubusercontent.com/f12mano/Genjustsu-Open-Source-Workflow/main/examples/previews/example-2.gif" width="33%" alt="Example 2" />
</p>

معاينة حية للنتائج قبل وبعد استبدال الشخصية وتوجيه الحركة بالمللي

---

## إيه ده وليه اتعمل

المشروع ده تحويل كامل لتجربة وتكنيك هندسة الفيديو العكسية من Higgsfield لبايبلاين سحابي حر يشتغل بضغطة زرار واحدة على Google Colab
بيحل مشكلتين أساسيتين واجهوا صناع المحتوى:
1. قيود وقت المنصات السحابية زي Hugging Face ZeroGPU اللي بتقفل بعد دقيقة وتمنع رندر الكليبات الكاملة
2. هلوسة موديلات الفيديو التوليدية زي Omni و Kling و Seedance لما تستلم كليب بدون قفل فيزيائي للأبعاد وحركة الوش

---

## أصل الفكرة وشكر واجب

الفكرة دي مبنية على الهندسة العكسية اللي عملها المطور Sirio Berati في مشروعه [Genjutsu Workflow](https://github.com/f12mano/Genjustsu-Open-Source-Workflow) وشرحها في الفيديو بتاعه [Dear Higgsfield, I reverse engineered you in 12 hours](https://www.youtube.com/watch?v=d6spCMcfkew)
هو أول واحد أثبت إن الدمج بين خريطة العمق وشبكة ملامح الوش مع الحفاظ على خلفية الأوضة الحقيقية هو السر اللي بيمنع موديلات توليد الفيديو من الهلوسة واختراع حركات عشوائية

### مقارنة معمارية بين الطريقتين

طريقة سيريو Genjutsu الأصلية:
- الميكانيزم: سيرفر محلي FastApi مع نفق Cloudflare وترحيل عمليات المعالجة عبر APIs سحابية مدفوعة على Replicate و Enhancor
- النقط الإيجابية: بتشتغل على أي لابتوب خفيف من غير كارت شاشة لأن الرندر الخارجي بيشيل الحمل / متكاملة مع خطوط الإنتاج البرمجية التلقائية
- التحديات: خطوات تسطيب بيئة بايثون محلية ومكتبات معالجة الصوت وفتح حسابات متعددة وإدارة مفاتيح API وشحن أرصدة مسبقة

طريقتنا One-Click Colab Runner:
- الميكانيزم: بيئة تشغيل موحدة على Google Colab بتشغل موديلات Meta و ByteDance و Google محلياً على كارت Nvidia T4 سحابي مباشر
- النقط الإيجابية: بضغطة زرار واحدة من غير أي تسطيب على جهازك ولا فتح حسابات خارجية ولا شحن أرصدة / بتطلعلك فيديو التوجيه الحركي كامل قدام عينك في دقيقة
- التحديات: جلسات كولاب المجانية مؤقتة وبتقفل عند الخمول / بتسلّمك الفيديو الإرشادي كملف جاهز ترفعه بنفسك للمنصة بدل الربط البرمجي التلقائي

---

## السجل الهندسي الميداني والدروس المستفادة

توثيق كامل للعيوب والحلول اللي تم حلها عملياً أثناء بناء البايبلاين:

1. عيب انهيار مكتبة Decord في كود C++ الداخلي:
- العرض: خطأ `Check failed: avcodec_send_packet >= 0 (-11 vs 0)` عند قراءة فيديوهات الموبايل أو تيك توك
- السبب: عدم توافق Decord مع معدل الفريمات المتغير وحزم الصوت المتداخلة
- الحل الحتمي: استبدال Decord بالكامل بدالة قراءة مخصصة مبنية على OpenCV مع ضمان أبعاد زوجية للفريمات

2. عيب عدم توافق مشغلات xFormers على كارت T4:
- العرض: خطأ `No operator found for memory_efficient_attention_forward`
- السبب: كيرنل xFormers غير متوافق مع أحجام التنسورات غير القياسية ومعمارية كارت T4 في Colab
- الحل الحتمي: تعطيل فلاج xFormers في الـ 5 موديولات وتطبيق باتش PyTorch SDPA الأصلي الموفر للذاكرة

3. عيب تسنين الحواف والهلوسة المكانية بالعمق الصرف:
- العرض: أومني بيخترع خلفية وإضاءة عشوائية لما يستلم فيديو الـ Depth لوحده في شاشة داكنة
- السبب: فقدان الموديل للحدود الفيزيائية والمكانية للغرفة الحقيقية
- الحل الحتمي: تطبيق معادلة سيريو الحرفية بدمج عمق الشخصية فوق خلفية الغرفة الطبيعية بدقة Meta SAM 2.1

4. قيد الوجوه المتعددة والزوايا الجانبية في MediaPipe:
- الحقيقة الميدانية: مكتبة MediaPipe Face Landmarker محسنة فقط لمتحدث واحد مواجه للكاميرا
- توجيه الاستخدام: استخدام بايبلاين الـ Face Mesh حصرياً لمقاطع الشخص الواحد المواجه / الاعتماد على DWPose أو Multi-Reference المباشر للمشاهد الجماعية

---

## خطوات التشغيل

1. افتح رابط النوت بوك المباشر في Google Colab
2. اتأكد إن نوع الحوسبة مضبوط على كارت T4 GPU من قائمة Runtime
3. شغّل كل الخلايا ورا بعض من Runtime وبعدها Run all أو اختصار `Ctrl + F9`
4. انزل للخلية رقم 4 وافتح رابط الويب المباشر اللي بينتهي بـ `.gradio.live`
5. ارفع الفيديو واختار المود المناسب ودوس Generate

---

## توثيق الموديلات الأصلية

- تقدير العمق الزمني: [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) من ByteDance
- عزل وتتبع الكتل في الفيديو: [SAM 2.1](https://github.com/facebookresearch/sam2) من Meta
- تتبع ملامح وشبكة الوجه: [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) من Google
