const express = require("express");
const bodyParser = require("body-parser");
// const request1 = require("request");
const https = require("https");
const app = express();

app.use(bodyParser.urlencoded({
  extended: true
}));
//This provides a path to our static files through a relative url (relative to the public folder).
app.use(express.static("Public"));

app.post("/", function(req, res) {
  const firstName = req.body.fName;
  const lastName = req.body.lName;
  const email = req.body.email;

// console.log(firstName)
// console.log(lastName)
// console.log(email)

  const data = {
    members: [{
      email_address: email,
      status: "subscribed",
      merge_fields: {
        FNAME: firstName,
        LNAME: lastName
      }
    }]
  };
  const jsonData = JSON.stringify(data);



const url = "https://us10.api.mailchimp.com/3.0/lists/52212a4cf4";


const options = {
  method: "POST",
  auth: "rutwij:b27bffac3a0df96102959c8d2032a589-us10"
}

const request = https.request(url, options, function(response) {

  if (response.statusCode === 200) {
    res.sendFile(__dirname + "/success.html");
  } else {
    res.sendFile(__dirname + "/failure.html");
  };

  response.on("data", function(data) {
    console.log(JSON.parse(data));
  });
});
request.write(jsonData);
request.end();
});

app.get("/", function(req, res) {
  res.sendFile(__dirname + "/signup.html");
});

app.post("/failure", function(req,res){
  res.redirect("/");
});

app.listen(process.env.PORT || 3000, function() {
  console.log("Server is running on port 3000");
});

//API KEY
//b27bffac3a0df96102959c8d2032a589-us10

//MY AUDIENCE ID
//52212a4cf4
