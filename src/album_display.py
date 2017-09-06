import os
import webbrowser
from random import randint
import numpy as np

icons_list = ["globe", "camera", "images", "leaf", "drop", "cloud", "paper-plane", "emoji-happy", "feather", "aircraft"]
# html templates paths
html_component_path = os.path.join("..", "display", "html_components")
album_header = os.path.join(html_component_path, "html_album_beginning.txt")
cluster_header = os.path.join(html_component_path, "html_cluster_beginning.txt")
page_footer = os.path.join(html_component_path, "html_page_ending.txt")
add_picture = os.path.join(html_component_path, "html_add_picture.txt")
add_picture_as_link = os.path.join(html_component_path, "html_add_picture_as_link.txt")
add_menu_page = os.path.join(html_component_path, "html_add_menu_page.txt")

def create_album_display(full_img_list, cluster_label_list, selected_img_list,
                         album_name, display=True, clustering=True):
    """
    
    :param full_img_list: paths list for all original images
    :param cluster_num_list: cluster label for each image in full_img_list 
    :param selected_img_list: the selected images that would be presented in the main page
    :param album_name: title name for the main html page (the album display)
    :param display: if true - opens the resulting html at the main page at the end of the process
    :param clustering: if true - creates html pages for the clustering results in addition to 
           the main page of the album gallery display
    :return: 
    """

    album_fn = album_name + ".html"
    # Create directory to save the album
    path = os.path.join("..", "albums")
    if not os.path.exists(path):
        os.makedirs(path)

    # Create new directory for the album
    album_dir_path = os.path.join(path, album_name)
    folder_num = 1
    while os.path.exists(album_dir_path):
        curr_album = album_name + str(folder_num)
        album_dir_path = os.path.join(path, curr_album)
        folder_num = folder_num + 1
    os.makedirs(album_dir_path)

    # Create html menu
    with open(add_menu_page, "r") as f:
        menu_page_str = f.read()
        menu_html = menu_page_str.replace("%%menu_page_link%%", album_fn)
        menu_html = menu_html.replace("%%menu_page_name%%", "Album")

    #####################################
    # Create clustering results display #
    #####################################
    cluster_pages_list = []
    cluster_fn_list = []
    if clustering:
        images = np.array(full_img_list)
        for ind, cluster_num in enumerate(np.unique(cluster_label_list)):
            # Create page header
            with open(cluster_header, "r") as f:
                cluster_html = f.read()
            cluster_html = cluster_html.replace("%%cluster_img%%", selected_img_list[ind])
            cluster_html = cluster_html.replace("%%cluster_img_ref%%", album_fn)

            # Add random icon to the title from icons_list
            on_left = randint(0, 1)
            title_icon = "icon-" + icons_list[randint(0, len(icons_list) - 1)]
            if on_left:
                cluster_html = cluster_html.replace("%%left_title_icon%%", title_icon)
            else:
                cluster_html = cluster_html.replace("%%right_title_icon%%", title_icon)

            # Create page content
            images_html = html_add_images(images[cluster_label_list == cluster_num])

            # Create page footer
            with open(page_footer, "r") as f:
                footer = f.read()

            # Create the cluster results page
            cluster_page = cluster_html + images_html + footer
            cluster_pages_list.append(cluster_page)
            cluster_str = "cluster" + str(ind) + ".html"
            cluster_fn_list.append(cluster_str)

        # Add clusters to menu
        for i in range(len(cluster_pages_list)):
            # Save the clusters pages
            cluster_str = "cluster " + str(i)
            cluster_fn = "cluster" + str(i) + ".html"

            menu_html = menu_html + menu_page_str
            menu_html = menu_html.replace("%%menu_page_link%%", cluster_fn)
            menu_html = menu_html.replace("%%menu_page_name%%", cluster_str)

    ########################
    # Create album display #
    ########################
    # Create page header
    with open(album_header, "r") as f:
        album_html = f.read()
    album_html = album_html.replace("%%album_title%%", album_name)

    # Add random icon to the title from icons_list
    on_left = randint(0,1)
    title_icon = "icon-" + icons_list[randint(0, len(icons_list)-1)]
    if on_left:
        album_html = album_html.replace("%%left_title_icon%%", title_icon)
    else:
        album_html = album_html.replace("%%right_title_icon%%", title_icon)

    # Create page content
    album_images_html = html_add_images(np.array(selected_img_list), cluster_fn_list)

    # Create page footer
    with open(page_footer, "r") as f:
        album_footer = f.read()

    # Create the album's main page
    album_main_page = album_html + album_images_html + album_footer

    # Save the album's main page
    album_path = os.path.join(album_dir_path, album_fn)

    album_main_page = album_main_page.replace("%%menu_content%%", menu_html)

    html_file = open(album_path, "w")
    html_file.write(album_main_page)
    html_file.close()

    if clustering:
        # Save the cluster pages
        for i in range(len(cluster_pages_list)):
            # Save the clusters pages
            cluster_fn = "cluster" + str(i) + ".html"
            cluster_path = os.path.join(album_dir_path, cluster_fn)

            cluster_page = cluster_pages_list[i].replace("%%menu_content%%", menu_html)

            if i == 0:
                # No previous page
                cluster_page = cluster_page.replace("%%prev_page_link%%", "#")
                cluster_page = cluster_page.replace("%%prev_page_icon%%", "")

            else:
                cluster_page = cluster_page.replace("%%prev_page_link%%", cluster_fn_list[i - 1])
                cluster_page = cluster_page.replace("%%prev_page_icon%%", "icon-triangle-left")

            if i == (len(cluster_pages_list)-1):
                # No next page
                cluster_page = cluster_page.replace("%%next_page_link%%", "#")
                cluster_page = cluster_page.replace("%%next_page_icon%%", "")
            else:
                cluster_page = cluster_page.replace("%%next_page_link%%", cluster_fn_list[i + 1])
                cluster_page = cluster_page.replace("%%next_page_icon%%", "icon-triangle-right")

            html_file = open(cluster_path, "w")
            html_file.write(cluster_page)
            html_file.close()

    if display:
        new = 2  # open in a new tab, if possible

        # open an HTML file
        webbrowser.open(album_path, new=new)


def html_add_images(img_list, img_link_str=None):
    html_str = ""
    for ind, img_path in enumerate(img_list):
        # img_str = "\"" + str(img_path) + "\""

        if img_link_str is None:
            with open(add_picture, "r") as f:
                data = f.read()
                data = data.replace("%%img_ref%%", str(img_path))
        else:
            with open(add_picture_as_link, "r") as f:
                data = f.read()
                data = data.replace("%%img_ref%%", img_link_str[ind])

        data = data.replace("%%img_path%%", str(img_path))
        html_str = html_str + data

    return html_str


