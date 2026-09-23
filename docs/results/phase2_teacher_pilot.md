# Phase 2b results: teacher pilot

Measured on 2026-09-23: the same 84 requests (21 intents × 4 varieties, 8 examples each) sent to both teachers; every kept example re-labelled blindly by the judge (Qwen3.5-35B-A3B (blind re-labelling of both runs)). The decision rule was fixed in `configs/sft/pilot_report.yaml` before the run.

Caveat: the judge is the candidate model, so it may slightly favour its own phrasing. No native speaker has reviewed any of this yet.

| | Qwen3.5-9B (Q4_K_M) | Qwen3.5-35B-A3B (UD-Q4_K_XL |
|---|---:|---:|
| Examples parsed | 672 | 672 |
| Kept after filters | 486 (72.3%) | 532 (79.2%) |
| Judge agrees with the intended label | 85.2% | 89.7% |
| **Usable** (kept and judge agrees) | **414 (61.6%)** | **477 (71.0%)** |
| Non-Gulf dialect rate (Gulf Arabic + mixed) | 16.0% | 2.9% |
| Generation speed | 173 tok/s | 50 tok/s |
| Seconds per usable example (generate + judge) | 0.73 | 1.56 |
| Estimated full run (20,000 usable) | 4.0 h | 8.7 h |

## Per variety

| Variety | Metric | Qwen3.5-9B (Q4_K_M) | Qwen3.5-35B-A3B (UD-Q4_K_XL |
|---|---|---:|---:|
| gulf_arabic | usable / parsed | 114/168 | 149/169 |
| gulf_arabic | non-Gulf rate | 12.0% | 1.8% |
| gulf_arabic | distinct-2 (variety) | 0.83 | 0.83 |
| gulf_arabic | avg words per message | 9.7 | 9.5 |
| english | usable / parsed | 164/168 | 152/168 |
| english | non-Gulf rate | 0.0% | 0.0% |
| english | distinct-2 (variety) | 0.84 | 0.87 |
| english | avg words per message | 11.8 | 10.6 |
| arabizi | usable / parsed | 121/168 | 118/168 |
| arabizi | non-Gulf rate | 0.0% | 0.0% |
| arabizi | distinct-2 (variety) | 0.66 | 0.74 |
| arabizi | avg words per message | 9.7 | 8.1 |
| mixed | usable / parsed | 15/168 | 58/167 |
| mixed | non-Gulf rate | 20.0% | 3.9% |
| mixed | distinct-2 (variety) | 0.80 | 0.91 |
| mixed | avg words per message | 11.6 | 9.3 |

## Rejection reasons

| Reason | Qwen3.5-9B (Q4_K_M) | Qwen3.5-35B-A3B (UD-Q4_K_XL |
|---|---:|---:|
| bad_fields | 0 | 3 |
| duplicate | 16 | 16 |
| message_lang_ar | 101 | 97 |
| message_lang_arabizi | 2 | 0 |
| message_lang_en | 38 | 9 |
| message_lang_mixed | 1 | 3 |
| near_duplicate | 1 | 6 |
| non_gulf_dialect | 25 | 6 |
| reply_lang_en | 2 | 0 |

## Most common judge disagreements

- **Qwen3.5-9B (Q4_K_M):** lost_or_stolen_card → card_not_working (7), missing_or_wrong_item → network_or_internet_issue (7), balance_or_statement → card_not_working (6), fees_and_charges → unrecognized_transaction (5), network_or_internet_issue → other (5), missing_or_wrong_item → order_status (5), sim_or_number → network_or_internet_issue (3), roaming → order_status (3)
- **Qwen3.5-35B-A3B (UD-Q4_K_XL:** other → handoff_to_human (4), card_not_working → lost_or_stolen_card (3), refund_request → missing_or_wrong_item (3), refund_request → failed_delivery (3), refund_request → order_status (3), unrecognized_transaction → other (2), fees_and_charges → unrecognized_transaction (2), loans_and_credit → card_not_working (2)

## Decision

Rule: the candidate wins if usable/parsed is at least 10% higher **or** its non-Gulf rate is at least 30% lower, **and** the full run fits in 10 h.

- Quality gain: +15.2%; non-Gulf drop: +81.9%; full run: 8.7 h (candidate) vs 4.0 h (baseline)
- → **Qwen3.5-35B-A3B (UD-Q4_K_XL** (better and fits the time budget)

## Random samples (usable examples)

### Qwen3.5-9B (Q4_K_M)

- `gulf_arabic` **change_delivery_details**: الحين ابغي اطلب لشنواين، مبش للجامعة.<br>→ نبينا العنوان الجديد فورا لتجنب التأخير.
- `gulf_arabic` **transfer_issue**: ابا، الحوالة مابغاتش زين ليش؟ ابغي الفاتورة بالحين!<br>→ نعتذر عن التأخير وسيدي، فريقنا باحلل الوضع ونبلغك فورا.
- `gulf_arabic` **card_not_working**: ليش هاي الفيزا مبعثورةش عند الآت أم؟ شو الحل؟<br>→ أهلا بك، نرجوك ما ترسل الرقم. راسلنا الصورة اللي طابعتك عشان نتأكد من وصول الكارت ونعافيك.
- `english` **plan_change**: Pls upgrade my data pack immediately! I need more 5G speed.<br>→ We can check your available upgrade options for you. Would you like me to show them?
- `english` **network_or_internet_issue**: My home internet is super slow, like 0 speed! Very angry customer here u guys fail.<br>→ We understand your frustration. Our team will run a diagnostic on your line right away.
- `english` **bill_inquiry**: bill came wrong amount again plz fix now<br>→ I am reviewing your recent charges right away. Please wait a few minutes while I check the details for you.
- `arabizi` **transfer_issue**: abi al7een ma3awwani shlon? abgha aro3 liya malak laish khadamatni?<br>→ ahlan bik, enta b3id minna. nt3ammi 3al wassel liya el-mob wal-ism bil-kamil ghadi n7allimk shu y9oul.
- `arabizi` **failed_delivery**: Shu wayed al7een? Order number 3399 sent back, abgha a7kal shu ykoun fih laish mashi 3indi<br>→ The parcel is being returned to our warehouse. We will send you a new one or refund your money.
- `arabizi` **balance_or_statement**: ashghal abi w ghadi n7e3 l branch, abgha a3ref shlon al7een awla mashi balansi wayed.<br>→ Please hold on while we access your records. The requested information will be forwarded to your phone number.
- `mixed` **refund_request**: شنو اللي صار للطلب؟ مكتوب delivered ومفيش شيء، اريد ريتريفت فوري.<br>→ سنتواصل معك لتأكيد التفاصيل ونبدأ عملية الرتريفت فورا.
- `mixed` **change_delivery_details**: Change delivery time من الساعة 2 ظهرا إلى 6 مساء because I'm in a meeting.<br>→ يمكنك تغيير الوقت إلى 6 مساء، هل تريد أن نعيد جدولة الطلب الآن؟
- `mixed` **network_or_internet_issue**: توقف الإي نت فجأة and the screen says no connection, ما نعرفو شو نفعل؟<br>→ يرجى تجربة كابل الشبكة مرة أخرى ثم أخبرنا إذا استمرت المشكلة لنرسل فني.

### Qwen3.5-35B-A3B (UD-Q4_K_XL

- `gulf_arabic` **transfer_issue**: شو صار بالمعاملة؟ ما وصلتش الزبون وانا ضايقة.<br>→ ما عليك، ارجو تزويدنا برقم التحويل لنفحص الوضع فورا.
- `gulf_arabic` **refund_request**: زبطت شيء غلط، ابغي استرجاع الفلوس.<br>→ حاضر، نبدأ عملية الاسترجاع وسنرسل لك تأكيدا قريبا.
- `gulf_arabic` **lost_or_stolen_card**: يا سلام، كرتي ضاع مني الحين وين اعمل؟<br>→ نحط لك طلب تجميد فوري. نرجو التوجه للفروع مع الهوية لاستخراج كرت جديد.
- `english` **lost_or_stolen_card**: U stole my card, need it frozen now.<br>→ Your card is frozen. Please visit a branch or call us for a new one.
- `english` **transfer_issue**: Why transfer failed? I paid for rent and landlord say no money received. Fix this asap!<br>→ We will look into why the payment was delayed. A specialist will contact you shortly with an update.
- `english` **bill_inquiry**: Lost my bill pdf, pls resend it urgently.<br>→ A new copy of your invoice has been emailed to you immediately. Please check your spam folder if you don't see it.
- `arabizi` **fees_and_charges**: abi afhem laish khadtoon feeze 25 dirham al7een w ma3rfoh?<br>→ la t7sen, shlon abthak lalfeeze wain tuftah?
- `arabizi` **sim_or_number**: abi esim jdid, wayed ya5dam mobaili.<br>→ zain, yajabek 2adam al-sim al-qadim w-tfaddal tji lil-fur3a.
- `arabizi` **order_status**: ma jayni el order! laish el driver ma yji?<br>→ el driver mshakl shwiya wabi y3awel yjibek. shlon ma t7t al3am?
- `mixed` **card_not_working**: بطاقتي blocked بالخطأ وأنا محتاجها لل ATM فورا.<br>→ نقدر نرفع الحظر فورا بعد التحقق من هويتك عبر التطبيق أو الاتصال بنا.
- `mixed` **order_status**: وين الباكج؟ ما وصلتني رسالة delivery time.<br>→ سأتحقق من تفاصيل الشحنة وأرسل لك التحديث فورا.
- `mixed` **fees_and_charges**: لماذا ظهر fee غريب على ال statement اليوم؟<br>→ سنقوم بفحص الحساب وإزالة أي خطأ إن وجد، انتظرنا قليلا.
