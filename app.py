from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/tree', methods=['POST'])
def suggestion():
    topic_list = []
    with open('static/HLDA.txt') as topics_file:
        for i in topics_file:
            topic_list.append(i.strip())
    # print(topic_list)

    required_lines = []
    tree_count_lev1 = 0
    tree_count_lev2 = 0
    level0_list = []
    level1_list = []
    level2_list = []
    for line in topic_list:
        if "level=0" in line:
            required_lines.append(line)
            level0_list.append(line)
        if "level=1" in line and tree_count_lev1 < 3:
            required_lines.append(line)
            level1_list.append(line)
            tree_count_lev1 += 1
            tree_count_lev2 = 0
        if "level=2" in line and tree_count_lev2 < 3:
            required_lines.append(line)
            level2_list.append(line)
            tree_count_lev2 += 1

    level0_box1 = level0_list[0]
    level1_box1 = level1_list[0]
    level1_box2 = level1_list[1]
    level1_box3 = level1_list[2]
    level2_box1 = level2_list[0]
    level2_box2 = level2_list[1]
    level2_box3 = level2_list[2]
    level2_box4 = level2_list[3]
    level2_box5 = level2_list[4]
    level2_box6 = level2_list[5]
    level2_box7 = level2_list[6]
    level2_box8 = level2_list[7]
    level2_box9 = level2_list[8]

    return render_template("result.html",
                           userData=request.form["search_text"],
                           level0box1=level0_box1,
                            level1box1= level1_box1,
                            level1box2= level1_box2,
                            level1box3= level1_box3,
                            level2box1= level2_box1,
                            level2box2= level2_box2,
                            level2box3= level2_box3,
                            level2box4= level2_box4,
                            level2box5= level2_box5,
                            level2box6= level2_box6,
                            level2box7= level2_box7,
                            level2box8= level2_box8,
                            level2box9= level2_box9
                           )


if __name__ == '__main__':
    app.run()
