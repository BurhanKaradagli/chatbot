from dotenv import load_dotenv
import streamlit as st
import os
import speech_recognition as sr
import google.generativeai as genai


load_dotenv()


st.set_page_config(page_title="Bitirme Projesi")


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

prompt_parts = [
  "input: Adana",
  "output: 17 Mayıs'ta Dj Mert Tugay, Jolly Joker Adana'da sahne alacak.\n24 Mayıs'ta Melek Mosso, yine Jolly Joker Adana'da performansını sergileyecek.\n25 Mayıs'ta Ceylan Ertem, \"Sezen Aksu Tribute\" adıyla bir konser verecek.\n26 Mayıs'ta \"Kanlı Nigar Müzikali\" ve Berkay'ın performansları olacak.\n27 Mayıs'ta \"Etekler ve Pantolonlar\" tiyatro gösterisi ve \"Ajda Forever\" adlı bir başka etkinlik düzenlenecek.\n28 Mayıs'ta Erol Evgin konseri planlanıyor.\n29 Mayıs'ta Sagopa Kajmer rap müziği severler için bir performans sunacak.\n30 Mayıs'ta Tahsin Hasoğlu sahne alacak.\n31 Mayıs'ta ise Haluk Levent konseri var.",
  "input: Adıyaman",
  "output: Mayıs 2024 için Adıyaman'da belirli eğlence etkinlikleriyle ilgili net bilgi bulamadım. Ancak, genel olarak Adıyaman ve çevresinde kültürel ve doğa ile ilgili etkinlikler düzenlenebiliyor. Örneğin, Adıyaman'da doğa yürüyüşleri, tarihi yerleri ziyaret etme gibi aktiviteler mevcut olabilir. Ayrıca, Adıyaman'ın tarihî ve kültürel zenginliklerini sunan Mount Nemrut gibi yerler de ziyaretçilere açıktır.\n\nBu tür etkinlikler genellikle yerel belediyeler, kültür merkezleri ya da turizm ofisleri tarafından düzenlenir. Etkinlik takvimi ve detayları için Adıyaman belediyesinin veya yerel turizm ofislerinin web sitelerini kontrol etmenizi öneririm.",
  "input: Afyonkarahisar",
  "output: Kocatepe Kültür ve Sanat Günleri: Bu etkinlik, yerel sanatçıların ve kültürel aktivitelerin yer aldığı bir festivaldir. Geleneksel halk dansları, tiyatro gösterileri ve el sanatları sergileri ile zenginleştirilecek.\n\nAfyonkarahisar Caz Festivali: Caz müziği severler için düzenlenen bu festival, yerel ve uluslararası caz sanatçılarını ağırlayacak. Canlı performanslar ve atölye çalışmaları da etkinliğin bir parçası olacak.\n\nAfyon Frig Vadisi Doğa Yürüyüşleri: Doğa tutkunları için Frig Vadisi'nde rehber eşliğinde doğa yürüyüşleri düzenleniyor. Bu etkinlikler, tarihi ve doğal güzellikleri keşfetmek isteyenler için ideal.\n\nYerel Pazarlar ve El Sanatları Fuarları: Ay boyunca çeşitli yerel pazarlar ve el sanatları fuarları da gerçekleşecek. Burada yöresel ürünleri ve el yapımı eserleri keşfedebilirsiniz.",
  "input: Ağrı",
  "output: ",
  "input: Aksaray",
  "output: ",
  "input: Amasya",
  "output: ",
  "input: Ankara",
  "output: David Garrett Trio Konseri: 5 Mayıs'ta Bilkent Üniversitesi Odeon'da ünlü kemancı David Garrett ve trio'su sahne alacak.\nTeoman Konseri: 22 Mayıs'ta Bilkent Odeon'da Teoman'ın konseri gerçekleşecek.\nGrup Abdal Konseri: 21 Mayıs'ta MEB Şura Salonu'nda Grup Abdal sahne alacak.\nFeridun Düzağaç Konseri: 22 Mayıs'ta 6:45 KK Ankara'da gerçekleşecek.\nYalın Konseri: 25 Mayıs'ta Congresium Ankara'da Yalın sahne alacak? (Songkick)?? (AllEvents.in)?? (AllEvents.in)?.",
  "input: Antalya",
  "output: Antalya Caz Festivali: 18-21 Mayıs tarihleri arasında düzenlenecek ve yerli, yabancı birçok caz sanatçısı sahne alacak.\nŞebnem Ferah Konseri: 23 Mayıs'ta Antalya Açıkhava Tiyatrosu'nda Şebnem Ferah sahne alacak.\nAntalya Film Festivali: Mayıs ayı boyunca çeşitli film gösterimleri ve söyleşiler düzenlenecek.\nAntalya Tiyatro Festivali: 15-25 Mayıs tarihleri arasında tiyatro oyunları ve performanslar sergilenecek? (Songkick)?.",
  "input: Ardahan",
  "output: ",
  "input: Artvin",
  "output: ",
  "input: Aydın",
  "output: ",
  "input: Balıkesir",
  "output: ",
  "input: Bartın",
  "output: ",
  "input: Batman",
  "output: ",
  "input: Bayburt",
  "output: ",
  "input: Bilecik",
  "output: ",
  "input: Bingöl",
  "output: ",
  "input: Bitlis",
  "output: ",
  "input: Bolu",
  "output: ",
  "input: Burdur",
  "output: ",
  "input: Bursa",
  "output: Kenan Doğulu Konseri: 20 Mayıs'ta Kültürpark Açıkhava Tiyatrosu'nda Kenan Doğulu sahne alacak.\nBursa Fotoğraf Festivali: Mayıs ayı boyunca fotoğraf sergileri ve atölye çalışmaları düzenlenecek.\nBursa Çikolata ve Kahve Festivali: 25-27 Mayıs tarihleri arasında gerçekleşecek ve çikolata, kahve tadımları ile atölyeler sunulacak.\nBursa Uluslararası Tiyatro Festivali: Çeşitli uluslararası tiyatro gruplarının katılımıyla Mayıs ayında farklı mekanlarda sahnelenecek? (Songkick)?.",
  "input: Çanakkale",
  "output: ",
  "input: Çankırı",
  "output: ",
  "input: Çorum",
  "output: ",
  "input: Denizli",
  "output: ",
  "input: Diyarbakır",
  "output: ",
  "input: Düzce",
  "output: ",
  "input: Edirne",
  "output: ",
  "input: Elazığ",
  "output: ",
  "input: Erzincan",
  "output: ",
  "input: Erzurum",
  "output: ",
  "input: Eskişehir",
  "output: ",
  "input: Gaziantep",
  "output: ",
  "input: Giresun",
  "output: ",
  "input: Gümüşhane",
  "output: ",
  "input: Hakkari",
  "output: ",
  "input: Hatay",
  "output: Hatayspor Maçı: 26 Mayıs 2024'te Yeni Hatay Stadyumu'nda Hatayspor ile Rizespor arasında bir futbol maçı düzenlenecek. Bu etkinlik, futbol severler için büyük bir heyecan kaynağı olacak? (Viagogo)?.\n\nHatay Gastronomi Festivali: Hatay'ın zengin mutfak kültürünü kutlamak için düzenlenen bu festivalde, yerel lezzetleri tatma ve yemek pişirme atölyelerine katılma fırsatı bulabilirsiniz. Tarihler ve detaylar için yerel kaynakları takip edebilirsiniz.\n\nKültür ve Sanat Etkinlikleri: Hatay'da Mayıs ayı boyunca çeşitli kültürel etkinlikler, sergiler ve sanat performansları düzenlenmektedir. Yerel kültür merkezleri ve belediyelerin web sitelerinden detaylı bilgi edinebilirsiniz.\n\nYerel Pazarlar ve El Sanatları Sergileri: Hatay'ın çeşitli bölgelerinde düzenlenen yerel pazarlar ve el sanatları sergileri, hem alışveriş yapmak hem de bölgenin kültürel zenginliklerini keşfetmek için harika fırsatlar sunar.",
  "input: Iğdır",
  "output: ",
  "input: Isparta",
  "output: ",
  "input: İstanbul",
  "output: Uluslararası İstanbul Tiyatro Festivali: Mayıs ve Haziran aylarında düzenlenen bu festival, çeşitli tiyatro gösterimlerine ev sahipliği yapacak. Festival boyunca yerli ve yabancı birçok tiyatro topluluğu sahne alacak? (Istanbeautiful)?.\n\nScorpions Konseri: 25 Mayıs'ta KüçükÇiftlik Park'ta gerçekleşecek bu konser, rock müziği sevenler için kaçırılmayacak bir etkinlik? (Concertful)?.\n\nDimash Kudaibergen Konseri: 24 Mayıs'ta Tüpraş Stadyumu'nda ünlü Kazak sanatçı Dimash Kudaibergen sahne alacak? (Songkick)?.\n\nİbrahim Maalouf Konseri: 25 Mayıs'ta Harbiye Cemil Topuzlu Açıkhava Tiyatrosu'nda gerçekleşecek bu konser, caz ve dünya müziği tutkunları için ideal? (Songkick)?.\n\nKoray Avcı Konseri: 26 Mayıs'ta Jolly Joker Vadistanbul'da sahne alacak olan Koray Avcı, hayranlarıyla buluşacak? (Songkick)?.\n\nZorlu PSM Etkinlikleri: Mayıs ayında Zorlu PSM'de Evrencan Gündüz (23 Mayıs), Still Corners (24 Mayıs) ve Gökhan Türkmen (29 Mayıs) gibi sanatçılar sahne alacak? (Songkick)?.\n\nHıdırellez Festivali: Baharın gelişini kutlamak için düzenlenen bu festivalde, geleneksel Türk müziği, dans gösterileri ve çeşitli etkinlikler yer alacak? (Istanbeautiful)?.",
  "input: İzmir",
  "output: Goran Bregovic Konseri: 24 Mayıs'ta İzmir Arena'da Goran Bregovic sahne alacak.\nMFÖ Konseri: 26 Mayıs'ta Kültürpark Açıkhava Tiyatrosu'nda MFÖ sahne alacak.\nİzmir Tiyatro Festivali: Mayıs ayı boyunca çeşitli tiyatro oyunları ve performanslar sergilenecek.\nİzmir Kitap Fuarı: 19-27 Mayıs tarihleri arasında gerçekleşecek ve birçok yazar, imza günleri ve söyleşilerle katılım gösterecek? (Songkick)?.",
  "input: Kahramanmaraş",
  "output: ",
  "input: Karabük",
  "output: ",
  "input: Karaman",
  "output: ",
  "input: Kars",
  "output: ",
  "input: Kastamonu",
  "output: ",
  "input: Kayseri",
  "output: ",
  "input: Kırıkkale",
  "output: ",
  "input: Kırklareli",
  "output: ",
  "input: Kırşehir",
  "output: ",
  "input: Kilis",
  "output: ",
  "input: Kocaeli",
  "output: ",
  "input: Konya",
  "output: ",
  "input: Kütahya",
  "output: ",
  "input: Malatya",
  "output: ",
  "input: Manisa",
  "output: ",
  "input: Mardin",
  "output: ",
  "input: Mersin",
  "output: ",
  "input: Muğla",
  "output: ",
  "input: Muş",
  "output: ",
  "input: Nevşehir",
  "output: ",
  "input: Niğde",
  "output: ",
  "input: Ordu",
  "output: ",
  "input: Osmaniye",
  "output: ",
  "input: Rize",
  "output: ",
  "input: Sakarya",
  "output: ",
  "input: Samsun",
  "output: ",
  "input: Siirt",
  "output: ",
  "input: Sinop",
  "output: ",
  "input: Sivas",
  "output: ",
  "input: Şanlıurfa",
  "output: Şanlıurfa Caz Festivali: Bu festival, caz müziği severler için düzenlenen önemli bir etkinlik. Yerel ve uluslararası caz sanatçılarının performanslarını izleme fırsatı bulabilirsiniz.\n\nGöbeklitepe Kültür ve Sanat Festivali: Tarihi Göbeklitepe alanında düzenlenen bu festivalde, yerel el sanatları, müzik performansları ve çeşitli kültürel etkinlikler yer alacak.\n\nBalıklıgöl Konserleri: Balıklıgöl bölgesinde çeşitli sanatçıların sahne alacağı konserler düzenlenecek. Bu etkinlikler, özellikle yaz aylarında açık hava konserleriyle ziyaretçileri çekiyor.\n\nŞanlıurfa Kebab ve Lezzet Festivali: Şanlıurfa'nın ünlü kebapları ve yöresel lezzetleri tanıtmak için düzenlenen bu festivalde, çeşitli yemek stantları ve tadım etkinlikleri olacak.",
  "input: Şırnak",
  "output: ",
  "input: Tekirdağ",
  "output: ",
  "input: Tokat",
  "output: ",
  "input: Trabzon",
  "output: ",
  "input: Tunceli",
  "output: ",
  "input: Uşak",
  "output: ",
  "input: Van",
  "output: ",
  "input: Yalova",
  "output: ",
  "input: Yozgat",
  "output: ",
  "input: Zonguldak",
  "output: ",
  "input: çanakkale",
  "output: ",
]

# Gemini'den cevap almak için fonksiyon
def gemini_cevap_al(soru):
    prompt_parts_with_input = prompt_parts + [f"input: {soru}"]
    response = model.generate_content(prompt_parts_with_input)
    return response.text

# Konuşmayı tanımlama fonksiyonu
def konusmayi_tani():
    r = sr.Recognizer()
    with sr.Microphone() as kaynak:
        st.write("Dinleniyor...")
        ses = r.listen(kaynak)
    try:
        st.write("İşleniyor...")
        kullanici_girdisi = r.recognize_google(ses, language="tr-TR")
        return kullanici_girdisi
    except sr.UnknownValueError:
        st.write("Üzgünüm, ne söylediğinizi anlayamadım.")
        return ""
    except sr.RequestError as e:
        st.write(f"Google Konuşma Tanıma servisinden sonuç alınamadı; {e}")
        return ""

# Ana Streamlit uygulaması
st.header("Bitirme Projesi Chatbot")
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

girdi_tipi = st.radio("Giriş Türü:", ("Metin", "Ses"))

if girdi_tipi == "Metin":
    metin_girdisi = st.text_input("Giriş: ", key="metin_girdisi")
    soru_sor = st.button("Sor")
    if soru_sor and metin_girdisi:
        cevap = gemini_cevap_al(metin_girdisi)
        st.session_state['chat_history'].append(("Siz", metin_girdisi))
        st.subheader("Cevap:")
        st.write(cevap)
        st.session_state['chat_history'].append(("Bot", cevap))
elif girdi_tipi == "Ses":
    st.write("Aşağıdaki düğmeye tıklayarak sorunuzu söyleyin.")
    if st.button("Kayıt"):
        metin_girdisi = konusmayi_tani()
        if metin_girdisi:
            cevap = gemini_cevap_al(metin_girdisi)
            st.session_state['chat_history'].append(("Siz", metin_girdisi))
            st.subheader("Cevap:")
            st.write(cevap)
            st.session_state['chat_history'].append(("Bot", cevap))

st.subheader("Sohbet Geçmişi")
for rol, metin in st.session_state['chat_history']:
    st.write(f"{rol}: {metin}")
