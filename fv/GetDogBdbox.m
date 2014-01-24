%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: GetDogBdbox
% Desc: get bounding box [x,y,wei,hei] from Stanford Dog Annotation
% Author: Zhang Kang
% Date: 2013/12/23
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GetDogBdbox: function description
function [ bdbox ] = GetDogBdbox( fileName )

% get the xpath mechanism into the workspace
import javax.xml.xpath.*
factory = XPathFactory.newInstance;
xpath = factory.newXPath;


xmlDoc = xmlread( fileName );

% xmlwrite( xmlDoc )


% compile and evaluate the XPath Expression
expression = xpath.compile('annotation/object/bndbox/xmin');
node = expression.evaluate(xmlDoc, XPathConstants.NODE);
xmin = str2double( node.getTextContent );

expression = xpath.compile('annotation/object/bndbox/ymin');
node = expression.evaluate(xmlDoc, XPathConstants.NODE);
ymin = str2double( node.getTextContent );

expression = xpath.compile('annotation/object/bndbox/xmax');
node = expression.evaluate(xmlDoc, XPathConstants.NODE);
xmax = str2double( node.getTextContent );

expression = xpath.compile('annotation/object/bndbox/ymax');
node = expression.evaluate(xmlDoc, XPathConstants.NODE);
ymax = str2double( node.getTextContent );

% transform bounding box format

bdbox = [ xmin, ymin, xmax - xmin, ymax - ymin ];


