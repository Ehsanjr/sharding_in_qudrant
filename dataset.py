
"""Generate deterministic Persian synthetic chunks."""

import random
import pandas as pd

TEAMS = {
    "hamta": "همتا",
    "karbordi": "کاربردی",
    "samin": "ثمین",
}

LEVELS = {
    "normal": "عادی",
    "confidential": "محرمانه",
}

SUBJECTS = [
    "احراز هویت کاربران",
    "مدیریت نقش ها",
    "پایش سرویس",
    "پشتیبان گیری داده ها",
    "ثبت رویدادهای امنیتی",
    "سیاست نگهداری اسناد",
    "مستندسازی رابط برنامه نویسی",
    "بررسی تغییرات کد",
    "مدیریت خطا",
    "بهینه سازی پایگاه داده",
    "همگام سازی منابع",
    "مدیریت گردش کار",
    "بررسی درخواست های کاربران",
    "بازیابی اطلاعات",
    "کنترل نسخه اسناد",
    "تحلیل کیفیت پاسخ",
    "برنامه ریزی انتشار",
    "تست یکپارچگی",
    "مدیریت دسترسی",
    "ساخت گزارش های عملیاتی",
]

ACTIONS = [
    "ارزیابی شده است",
    "در مخزن دانش ثبت شده است",
    "برای بازبینی آماده است",
    "با سیاست های جاری تطبیق داده شده است",
    "نیازمند بررسی عملکرد است",
    "در چرخه تست قرار گرفته است",
    "با سرویس های وابسته هماهنگ شده است",
    "برای نسخه بعدی اصلاح شده است",
]

DETAILS = [
    "شاخص تاخیر درخواست ها",
    "محدودیت دسترسی گروه ها",
    "برنامه بازیابی سرویس",
    "ثبت تغییرات و تاییدیه ها",
    "کنترل کیفیت خروجی",
    "فرایند ارجاع درخواست",
    "به روزرسانی دانش سازمانی",
    "اندازه گیری نرخ خطا",
]


def make_dataset(n=10000, seed=2026):
    rng = random.Random(seed)
    rows = []
    teams = list(TEAMS)

    for i in range(n):
        team = teams[i % 3]

        level = (
            "confidential"
            if (i * 17 + 9) % 5 < 2
            else "normal"
        )

        subject = SUBJECTS[rng.randrange(len(SUBJECTS))]
        action = ACTIONS[rng.randrange(len(ACTIONS))]
        detail = DETAILS[rng.randrange(len(DETAILS))]

        text = (
            f"گزارش شماره {i + 1} از تیم {TEAMS[team]} "
            f"در سطح {LEVELS[level]}: "
            f"موضوع {subject} {action}. "
            f"در این فعالیت {detail} اهمیت دارد. "
            f"نتیجه برای پیگیری، مقایسه و جستجوی دانش "
            f"ثبت شده است."
        )

        rows.append({
            "id": i + 1,
            "text": text,
            "team": team,
            "classification": level,
        })

    return pd.DataFrame(rows)
